"""Precisão e recall por limiar na validação, com cobertura parcial preservada."""
from __future__ import annotations
from collections import Counter
import gzip,json
from pathlib import Path
import numpy as np
import diag11_core as core
from src import politica_alertas as policy
from src.validacao_alertas import corresponder_objetos
from learn21_protocol import SPEC

def profile(records,cache,threshold,spec=SPEC,details=False):
    totals=np.zeros(3,dtype=np.int64);alert_totals=np.zeros(3,dtype=np.int64)
    per=np.zeros((12,3),dtype=np.int64);support=np.zeros(12,dtype=np.int64);high=missing_high=unknown=0
    images=[]
    for r in records:
        if r['role'] not in {'home_val','val_nova'}:raise ValueError('Somente validações de desenvolvimento podem selecionar pesos e limiares.')
        gt=core.with_policy([g for g in r['gt'] if r['known_mask']&(1<<g['classe_id'])],r['width'],r['height'],policy)
        predictions=core.with_policy([p for p in cache[r['id']] if p['confianca']>=threshold],r['width'],r['height'],policy)
        pred=[p for p in predictions if r['known_mask']&(1<<p['classe_id'])];unknown+=len(predictions)-len(pred)
        pairs,missing,extras=corresponder_objetos(gt,pred)
        counts=np.array([len(pairs),len(missing),len(extras)],dtype=np.int64);totals+=counts
        for g in gt:support[g['classe_id']]+=1
        for gi,pi,_ in pairs:per[gt[gi]['classe_id'],0]+=1
        for gi in missing:per[gt[gi]['classe_id'],1]+=1
        for pi in extras:per[pred[pi]['classe_id'],2]+=1
        events=[(gt[gi],pred[pi]) for gi,pi,_ in pairs]+[(gt[gi],None) for gi in missing]+[(None,pred[pi]) for pi in extras]
        ac=[0,0,0]
        for g,p in events:
            ga,pa=core.alert_on(g),core.alert_on(p)
            ac[0]+=ga and pa;ac[1]+=ga and not pa;ac[2]+=not ga and pa
            high+=bool(g and g['prioridade']=='alto');missing_high+=bool(g and g['prioridade']=='alto' and not pa)
        alert_totals+=ac
        if details:images.append({'id':r['id'],'coorte':r['coorte'],'confianca':threshold,'tp_detector':int(counts[0]),'fn_detector':int(counts[1]),'fp_detector':int(counts[2]),
            'tp_alerta':ac[0],'fn_alerta':ac[1],'fp_alerta':ac[2],
            'mensagens':'\n'.join(policy.gerar_mensagem(p,'medio')+(' Classe sem cobertura de referência.' if not r['known_mask']&(1<<p['classe_id']) else '') for p in predictions if core.alert_on(p))})
    det=core.alert_metrics(*totals);alerts=core.alert_metrics(*alert_totals)
    score=min((det['precision'] or 0)/spec['target_precision'],(det['recall'] or 0)/spec['target_recall'])
    rows=[{'classe_id':cid,'classe':name,'gt_avaliaveis':int(support[cid]),'suporte_reduzido':bool(support[cid]<30),**core.alert_metrics(*per[cid])} for cid,name in enumerate(core.NAMES)]
    return {'threshold':threshold,'images':len(records),'detector':det,'alertas':alerts,'alta_ref':high,'alta_sem_alerta':missing_high,
        'predicoes_sem_cobertura':unknown,'target_met':bool(det['precision'] is not None and det['recall'] is not None and det['precision']>=spec['target_precision'] and det['recall']>=spec['target_recall']),
        'fitness':float(score+1e-6*(det['f1'] or 0)),'classes':rows,'image_rows':images}

def choose(profiles):
    if not profiles:raise ValueError('Grade de validação vazia.')
    return max(profiles,key=lambda p:(p['fitness'],-p['threshold']))

def capture(model,records,device,spec=SPEC):
    if list(model.names.values())!=core.NAMES:raise ValueError('O modelo não tem as 12 classes esperadas, na ordem correta.')
    cache={};saturated=0
    for i,r in enumerate(records,1):
        if r['role'] not in {'home_val','val_nova'}:raise ValueError('Tentativa de consultar treino ou teste na seleção.')
        result=model.predict(str(r['image']),conf=spec['inference_conf'],imgsz=spec['imgsz'],rect=True,
            iou=spec['nms_iou'],max_det=spec['max_det'],half=False,batch=1,device=device,verbose=False,save=False)[0]
        if Path(result.path).resolve()!=Path(r['image']).resolve() or tuple(result.orig_shape)!=(r['height'],r['width']):raise ValueError('Previsão retornou outra imagem.')
        boxes=result.boxes
        preds=[] if boxes is None else [{'classe_id':int(cid),'classe':core.NAMES[int(cid)],'bbox':[float(x) for x in box],'confianca':float(conf)}
            for cid,box,conf in zip(boxes.cls.cpu().numpy(),boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy())]
        from conf14_evaluation import validate_predictions
        validate_predictions(preds,r);cache[r['id']]=preds;saturated+=len(preds)==spec['max_det']
        if i%250==0:print(f'Validação {r["coorte"]}: {i}/{len(records)}',flush=True)
    return cache,saturated

def evaluate(model,cohorts,device,spec=SPEC,output=None):
    caches={};profiles={};saturation={}
    for cohort,records in cohorts.items():
        caches[cohort],saturation[cohort]=capture(model,records,device,spec)
        if cohort==spec['primary_cohort']:
            profiles[cohort]=[profile(records,caches[cohort],c,spec) for c in spec['thresholds']]
    selected=choose(profiles[spec['primary_cohort']]);threshold=selected['threshold']
    summaries={c:profile(records,caches[c],threshold,spec,details=output is not None) for c,records in cohorts.items()}
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        core.save_json(output/'grade_limiares.json',profiles[spec['primary_cohort']])
        for cohort,records in cohorts.items():
            target=output/cohort;target.mkdir(exist_ok=True)
            core.save_json(target/'metricas.json',summaries[cohort]);core.save_csv(target/'metricas_por_classe.csv',summaries[cohort]['classes'])
            core.save_csv(target/'metricas_por_imagem.csv',summaries[cohort]['image_rows'])
            with gzip.open(target/'predicoes.jsonl.gz','wt',encoding='utf-8',compresslevel=4) as f:
                for r in records:f.write(json.dumps({'id':r['id'],'image_sha256':r['sha256'],'label_sha256':r['label_sha256'],'predicoes':caches[cohort][r['id']]},ensure_ascii=False)+'\n')
        core.save_json(output/'limite_captura.json',{'confianca_captura':spec['inference_conf'],'max_det':spec['max_det'],'imagens_atingiram_max_det':saturation})
    return selected,summaries,profiles[spec['primary_cohort']]
