"""Mesmo associador e política V2 históricos, com seleção exclusivamente em VAL."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import gzip,json,math
import numpy as np
import home24_common as c
from src.validacao_alertas import corresponder_objetos
from src import politica_alertas as policy

def metrics(tp,fn,fp):
    tp,fn,fp=map(int,(tp,fn,fp))
    return {'tp':tp,'fn':fn,'fp':fp,'precision':tp/(tp+fp) if tp+fp else None,
      'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else None}
def alert(obj):return bool(obj and obj['prioridade'] in {'medio','alto'})
def with_policy(obj,r):
    b=obj['bbox']
    if b[2]<=b[0] or b[3]<=b[1]:
        return {**obj,'prioridade':'sem_prioridade','geometria_degenerada':True}
    return {**obj,**policy.avaliar_bbox(obj['classe'],b,r['width'],r['height'],versao='v2')}
def profile(rows,cache,threshold,ids=None,details=True):
    allowed=set(range(12) if ids is None else ids);total=np.zeros(3,dtype=np.int64)
    per=np.zeros((12,3),dtype=np.int64);at=np.zeros(3,dtype=np.int64);high=high_missing=0;image_rows=[]
    for r in rows:
        gt=[with_policy(g,r) for g in r['gt'] if g['classe_id'] in allowed]
        pred=[with_policy(p,r) for p in cache[r['id']] if p['confianca']>=threshold and p['classe_id'] in allowed]
        pairs,missing,extras=corresponder_objetos(gt,pred);counts=np.asarray([len(pairs),len(missing),len(extras)]);total+=counts
        for gi,_,_ in pairs:per[gt[gi]['classe_id'],0]+=1
        for gi in missing:per[gt[gi]['classe_id'],1]+=1
        for pi in extras:per[pred[pi]['classe_id'],2]+=1
        events=[(gt[i],pred[j]) for i,j,_ in pairs]+[(gt[i],None) for i in missing]+[(None,pred[j]) for j in extras]
        ac=[sum(alert(g) and alert(p) for g,p in events),sum(alert(g) and not alert(p) for g,p in events),sum(not alert(g) and alert(p) for g,p in events)]
        hi=sum(g['prioridade']=='alto' for g in gt);hm=sum(bool(g and g['prioridade']=='alto' and not alert(p)) for g,p in events)
        at+=ac;high+=hi;high_missing+=hm
        if details:image_rows.append({'id':r['id'],'tp':int(counts[0]),'fn':int(counts[1]),'fp':int(counts[2]),
          'tp_alerta':ac[0],'fn_alerta':ac[1],'fp_alerta':ac[2],'alta_ref':hi,'alta_sem_alerta':hm,
          'mensagens':'\n'.join(policy.gerar_mensagem(p,'medio') for p in pred if alert(p))})
    det=metrics(*total);fitness=min((det['precision'] or 0)/.8,(det['recall'] or 0)/.7)+1e-6*(det['f1'] or 0)
    return {'images':len(rows),'threshold':threshold,'class_ids':sorted(allowed),'detector':det,'alertas':metrics(*at),
      'alta_ref':high,'alta_sem_alerta':high_missing,'fitness':fitness,
      'target_met':bool(det['precision'] is not None and det['precision']>=.8 and det['recall'] is not None and det['recall']>=.7),
      'classes':[{'class_id':i,'class':c.NAMES[i],'gt':int(per[i,0]+per[i,1]),**metrics(*per[i])} for i in sorted(allowed)],'image_rows':image_rows}
def choose(grid):return max(grid,key=lambda p:(p['fitness'],-p['threshold']))

def validate_pred(values,row):
    for p in values:
        b=p['bbox'];v=p['confianca'];cid=p['classe_id']
        if cid not in range(12) or len(b)!=4 or not all(math.isfinite(x) for x in b+[v]) or not 0<=v<=1 or b[2]<b[0] or b[3]<b[1]:
            raise ValueError('Previsão inválida em '+row['id'])
        if min(b)<-.05 or b[2]>row['width']+.05 or b[3]>row['height']+.05:raise ValueError('Dimensões de previsão incompatíveis em '+row['id'])
def predict_yolo(model,rows,imgsz=640,device=0):
    if list(model.names.values())!=c.NAMES:raise ValueError('Ordem das classes YOLO alterada.')
    cache={}
    for r in rows:
        out=model.predict(str(r['image']),imgsz=imgsz,conf=.01,iou=.7,max_det=300,quantize=32,device=device,verbose=False,save=False)[0]
        if Path(out.path).resolve()!=Path(r['image']).resolve() or tuple(out.orig_shape)!=(r['height'],r['width']):raise ValueError('YOLO retornou outra imagem.')
        boxes=out.boxes
        values=[] if boxes is None else [{'classe_id':int(cid),'classe':c.NAMES[int(cid)],'bbox':[float(x) for x in box],'confianca':float(conf)}
            for cid,box,conf in zip(boxes.cls.cpu().numpy(),boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy())]
        validate_pred(values,r);cache[r['id']]=values
    return cache
def predict_rf(model,rows,batch_size=4):
    cache={}
    for start in range(0,len(rows),batch_size):
        block=rows[start:start+batch_size]
        result=model.predict([str(r['image']) for r in block],threshold=.01)
        if not isinstance(result,list):result=[result]
        if len(result)!=len(block):raise ValueError('RF-DETR retornou quantidade incorreta de imagens.')
        for r,out in zip(block,result):
            values=[{'classe_id':int(cid),'classe':c.NAMES[int(cid)],'bbox':[float(x) for x in box],'confianca':float(conf)}
              for cid,box,conf in zip(out.class_id,out.xyxy,out.confidence)]
            validate_pred(values,r);cache[r['id']]=values
    return cache

def obtain_cache(rows,output,identity,predictor,allow_role,checkpoint=None):
    if any(r['role']!=allow_role for r in rows):raise ValueError('Papel proibido na inferência: '+allow_role)
    req={'identity':identity,'input':[{k:r[k] for k in ['id','sha256','label_sha256','width','height']} for r in rows],
      'capture_conf':.01,'role':allow_role,'max_det':300};key=c.digest(req)
    base=Path(output)/'cache'/key[:24];base.mkdir(parents=True,exist_ok=True);cache={};diagnostics=[]
    for start in range(0,len(rows),32):
        block=rows[start:start+32];p=base/f'bloco_{start//32:04d}.json.gz'
        if p.is_file():
            with gzip.open(p,'rt') as f:payload=json.load(f)
            if payload['key']!=key or payload['data_sha256']!=c.digest(payload['data']) or list(payload['data'])!=[r['id'] for r in block]:raise ValueError('Cache de outra entrada ou alterado.')
        else:
            data=predictor(block)
            if list(data)!=[r['id'] for r in block]:raise ValueError('As previsões não correspondem ao bloco.')
            for r in block:validate_pred(data[r['id']],r)
            payload={'key':key,'data':data,'data_sha256':c.digest(data)};tmp=p.with_suffix('.tmp')
            with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=4) as f:json.dump(payload,f,ensure_ascii=False,allow_nan=False)
            tmp.replace(p)
        cache.update(payload['data'])
        print(f'Inferência {allow_role}: {start+len(block)}/{len(rows)}',flush=True)
        if checkpoint:checkpoint()
    for r in rows:
        values=cache[r['id']];diagnostics.append({'id':r['id'],'predictions':len(values),'native_limit_reached':len(values)>=300,
           'zero_area':sum(p['bbox'][2]==p['bbox'][0] or p['bbox'][3]==p['bbox'][1] for p in values)})
    c.save(base/'manifesto_cache.json',{'key':key,'request':req,'complete':True,'diagnostics':diagnostics})
    return cache

def development(rows,cache,output):
    if any(r['role']!='val' for r in rows):raise ValueError('Seleção exige somente VAL.')
    grid=[profile(rows,cache,t,details=False) for t in c.SPEC['thresholds']];selected=choose(grid)
    primary=profile(rows,cache,selected['threshold']);secondary=profile(rows,cache,selected['threshold'],c.ALERT_IDS)
    c.save(Path(output)/'grade_limiares.json',grid);save_profiles(output,primary,secondary)
    return primary,secondary
def save_profiles(output,primary,secondary):
    output=Path(output);c.save(output/'metricas_12_classes.json',primary);c.save(output/'metricas_6_alerta.json',secondary)
    c.csv_save(output/'classes.csv',primary['classes']);c.csv_save(output/'imagens.csv',primary['image_rows'])

def bootstrap(profiles,replicates=None,seed=None):
    replicates=replicates or c.SPEC['bootstrap_replicates'];seed=seed or c.SPEC['seed'];names=list(profiles)
    ids=[[r['id'] for r in profiles[name]['image_rows']] for name in names]
    if not ids[0] or any(v!=ids[0] for v in ids):raise ValueError('Bootstrap pareado exige as mesmas imagens, na mesma ordem.')
    mats={name:np.asarray([[r[k] for k in ['tp','fn','fp','tp_alerta','fn_alerta','fp_alerta','alta_ref','alta_sem_alerta']] for r in profiles[name]['image_rows']],dtype=np.int64) for name in names}
    keys=['precision','recall','f1','alert_precision','alert_recall','alert_f1','high_missing_rate']
    samples={name:{k:[] for k in keys} for name in names};rng=np.random.default_rng(seed);n=len(ids[0])
    for _ in range(replicates):
        indices=rng.integers(0,n,n)
        for name in names:
            v=mats[name][indices].sum(axis=0);d=metrics(*v[:3]);a=metrics(*v[3:6]);vals=[d['precision'],d['recall'],d['f1'],a['precision'],a['recall'],a['f1'],v[7]/v[6] if v[6] else None]
            for k,value in zip(keys,vals):samples[name][k].append(value)
    interval=lambda values:np.quantile([v for v in values if v is not None],[.025,.975]).tolist() if any(v is not None for v in values) else None
    result={'unit':'imagem','groups':n,'replicates':replicates,'seed':seed,'limite':'Sem agrupamento validado de cenas Home; correlação entre imagens pode estreitar os intervalos.',
      'interval_95':{name:{k:interval(v) for k,v in data.items()} for name,data in samples.items()},'paired_deltas':{}}
    if len(names)>1:
        reference=names[0]
        for name in names[1:]:
            result['paired_deltas'][name+'_minus_'+reference]={k:interval([a-b if a is not None and b is not None else None for a,b in zip(samples[name][k],samples[reference][k])]) for k in keys}
    return result
