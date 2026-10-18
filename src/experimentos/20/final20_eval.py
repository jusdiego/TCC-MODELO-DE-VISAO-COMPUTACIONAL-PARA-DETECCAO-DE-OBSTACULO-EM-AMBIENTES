"""Inferência com retomada e avaliação explícita da reserva final autorizada."""
from __future__ import annotations
import gzip,json
from pathlib import Path
import diag11_core as core
import conf14_evaluation as e14
from final20_data import COHORT,ROLE,MODELS,digest,save_gz

def validate_records(records,plan):
    if len(records)!=plan['images'] or len({r['id'] for r in records})!=len(records):
        raise ValueError('Avaliação exige todos os IDs únicos da reserva.')
    if any(r['role']!=ROLE or r['coorte']!=COHORT for r in records):raise ValueError('Papel proibido nesta avaliação final.')

def predict(model,record,device,plan):
    if record['role']!=ROLE or record['coorte']!=COHORT:raise ValueError('Inferência somente na reserva autorizada.')
    args={k:v for k,v in plan['inference'].items() if k!='ultralytics'}
    result=model.predict(str(record['image']),**args,device=device,verbose=False,save=False)[0]
    if Path(result.path).resolve()!=Path(record['image']).resolve() or tuple(result.orig_shape)!=(record['height'],record['width']):
        raise ValueError('O modelo retornou uma imagem ou dimensão diferente da entrada.')
    boxes=result.boxes
    if boxes is None:return []
    return [{'classe_id':int(c),'classe':core.NAMES[int(c)],'bbox':[float(x) for x in box],'confianca':float(score)}
            for c,box,score in zip(boxes.cls.cpu().numpy(),boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy())]

def cache_request(records,model_name,weight_sha,environment,plan):
    validate_records(records,plan)
    if model_name not in MODELS or weight_sha!=plan['weight_sha256'][model_name]:raise ValueError('Peso fora do protocolo final.')
    signatures=[{k:r[k] for k in ['id','sha256','label_sha256','coverage_sha256','width','height','known_mask','role']} for r in records]
    return {'versao':'20-cache-reserva-v1','modelo':model_name,'peso_sha256':weight_sha,'parametros':plan['inference'],
        'protocolo_sha256':digest(plan),'ambiente':environment,'entradas_sha256':digest(signatures),'imagens':len(records)}

def obtain_cache(model,records,model_name,weight_sha,environment,device,plan,directory,checkpoint=None,predict_fn=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    request=cache_request(records,model_name,weight_sha,environment,plan);request_sha=digest(request)
    block_size=25;cache={};reused=0;generated=0;geometry=[]
    for start in range(0,len(records),block_size):
        block=records[start:start+block_size];path=directory/f'lote_{start//block_size:04d}.json.gz'
        if path.is_file():
            payload=json.loads(gzip.decompress(path.read_bytes()))
            if payload.get('request_sha256')!=request_sha or payload.get('rows_sha256')!=digest(payload.get('rows')):
                raise ValueError('Lote de previsões alterado ou de outro contexto: '+path.name)
            values=payload['rows'];reused+=len(values)
            if [v['id'] for v in values]!=[r['id'] for r in block]:raise ValueError('Lote com IDs ou ordem incompatíveis.')
        else:
            values=[]
            for record in block:
                predictions=(predict_fn or predict)(model,record,device,plan)
                e14.validate_predictions(predictions,record)
                if len(predictions)>plan['inference']['max_det']:raise ValueError('Cache excede max_det congelado.')
                values.append({'id':record['id'],'image_sha256':record['sha256'],'label_sha256':record['label_sha256'],
                    'coverage_sha256':record['coverage_sha256'],'predicoes':predictions})
            save_gz(path,{'request_sha256':request_sha,'rows_sha256':digest(values),'rows':values});generated+=len(values)
        for value,record in zip(values,block):
            if any(value[key]!=record[field] for key,field in [('image_sha256','sha256'),('label_sha256','label_sha256'),('coverage_sha256','coverage_sha256')]):
                raise ValueError('Previsões pertencem a outra imagem ou referência.')
            diagnostic=e14.validate_predictions(value['predicoes'],record)
            geometry.append({'id':record['id'],'diagnostico':diagnostic})
            if len(value['predicoes'])>plan['inference']['max_det']:raise ValueError('Cache excede max_det.')
            cache[record['id']]=value['predicoes']
        done=start+len(block)
        core.save_json(directory/'progresso_cache.json',{'request_sha256':request_sha,'imagens_concluidas':done,'imagens_previstas':len(records)})
        print(f'{model_name}: {done}/{len(records)} imagens; {reused} reutilizadas',flush=True)
        if checkpoint and (done%100==0 or done==len(records)):checkpoint()
    if set(cache)!={r['id'] for r in records}:raise ValueError('Cache final incompleto.')
    unusual=[v for v in geometry if v['diagnostico']['area_zero'] or v['diagnostico']['fora_da_imagem']]
    counts={k:sum(v['diagnostico'][k] for v in geometry) for k in ['predicoes','area_zero','fora_da_imagem']}
    geom={'propostas_removidas':0,'coordenadas_alteradas':False,'totais':counts,'casos':unusual,
        'imagens_atingiram_max_det':sum(len(v)==plan['inference']['max_det'] for v in cache.values())}
    core.save_json(directory/'diagnostico_geometria.json',geom)
    meta={'solicitacao':request,'request_sha256':request_sha,'concluido':True,'imagens':len(cache),
        'imagens_reutilizadas_nesta_execucao':reused,'imagens_inferidas_nesta_execucao':generated,
        'limite':'Propostas pós-NMS em conf=0,001 e max_det=300; sem logits brutos.',
        'arquivos':[{ 'arquivo':p.name,'sha256':core.sha256_file(p)} for p in sorted(directory.glob('lote_*.json.gz'))]}
    core.save_json(directory/'cache_manifesto.json',meta)
    return cache,meta

def analyse(record,predictions,model_name,plan,policy,matcher):
    if record['role']!=ROLE or record['coorte']!=COHORT:raise ValueError('Avaliação somente da reserva final.')
    known=record['known_mask'];conf=plan['alert_confidence']
    gt=core.with_policy([g for g in record['gt'] if known&(1<<g['classe_id'])],record['width'],record['height'],policy)
    low=[p for p in predictions if known&(1<<p['classe_id'])]
    all_pred=core.with_policy([p for p in predictions if p['confianca']>=conf],record['width'],record['height'],policy)
    pred=[p for p in all_pred if known&(1<<p['classe_id'])]
    pairs,missing,extras=matcher(gt,pred)
    events=[e14.event(record,'correspondencia',gt[gi],pred[pi],float(overlap),low,conf) for gi,pi,overlap in pairs]
    events += [e14.event(record,'objeto_nao_detectado',gt[gi],None,None,low,conf) for gi in missing]
    events += [e14.event(record,'previsao_nao_associada',None,pred[pi],None,low,conf) for pi in extras]
    for event in events:event['modelo']=model_name
    messages=[]
    for obj in all_pred:
        if core.alert_on(obj):
            message=policy.gerar_mensagem(obj,'medio')
            if not known&(1<<obj['classe_id']):message+=' Classe sem cobertura de referência nesta imagem.'
            messages.append(message)
    if any(not isinstance(m,str) or not m.strip() for m in messages):raise ValueError('Alerta sem mensagem.')
    image={'coorte':COHORT,'imagem':record['name'],'id':record['id'],'modelo':model_name,'limiar_confianca':conf,
        'tp_alerta':sum(e['ha_alerta_ref'] and e['ha_alerta_pred'] for e in events),
        'fn_alerta':sum(e['ha_alerta_ref'] and not e['ha_alerta_pred'] for e in events),
        'fp_alerta':sum(not e['ha_alerta_ref'] and e['ha_alerta_pred'] for e in events),
        'alta_ref':sum(g['prioridade']=='alto' for g in gt),
        'alta_sem_alerta':sum(e['prioridade_ref']=='alto' and not e['ha_alerta_pred'] for e in events),
        'tp_detector':len(pairs),'fn_detector':len(missing),'fp_detector':len(extras),
        'gt_avaliaveis':len(gt),'gt_ignorados':len(record['gt'])-len(gt),
        'predicoes_ignoradas_conf':len(all_pred)-len(pred),
        'cobertura_completa_alertas':known&core.ALERT_MASK==core.ALERT_MASK,
        'ha_alerta_ref':any(core.alert_on(g) for g in gt),'ha_alerta_pred':any(core.alert_on(p) for p in pred),
        'mensagem_alerta':'\n'.join(messages)}
    return image,events

def evaluate(records,cache,model_name,plan,policy,matcher,output):
    validate_records(records,plan)
    if model_name not in MODELS or set(cache)!={r['id'] for r in records}:raise ValueError('Modelo ou IDs fora do protocolo.')
    images=[];events=[]
    for record in records:
        e14.validate_predictions(cache[record['id']],record)
        image,part=analyse(record,cache[record['id']],model_name,plan,policy,matcher)
        images.append(image);events.extend(part)
    summary,classes=e14.summarise(images,events,plan['alert_confidence'])
    summary.update(modelo=model_name,papel=ROLE,reserva_final=True,treinamento_executado=False)
    summary['predicoes_sem_cobertura_no_limiar']=summary['predicoes_ignoradas_conf']
    for row in classes:
        cid=row['classe_id'];part=[e for e in events if e['classe_id']==cid]
        row['taxa_alta_sem_alerta']=core.ratio(row['alta_sem_alerta'],row['alta_ref'])
        row['imagens_com_cobertura']=sum(bool(r['known_mask']&(1<<cid)) for r in records)
        detector=core.alert_metrics(sum(e['evento']=='correspondencia' for e in part),
            sum(e['evento']=='objeto_nao_detectado' for e in part),sum(e['evento']=='previsao_nao_associada' for e in part))
        row.update({'detector_'+k:v for k,v in detector.items()})
    if summary['alertas_por_objeto']['tp']+summary['alertas_por_objeto']['fn']!=sum(core.alert_on(g) for r in records for g in core.with_policy([g for g in r['gt'] if r['known_mask']&(1<<g['classe_id'])],r['width'],r['height'],policy)):
        raise ValueError('Denominador de alertas GT inconsistente.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    core.save_json(output/'metricas.json',summary)
    core.save_csv(output/'metricas_por_classe.csv',classes)
    core.save_csv(output/'alertas_por_imagem.csv',images)
    core.save_csv(output/'avaliacao_objetos.csv',events)
    result={'summary':summary,'classes':classes,'images':images,'events':events}
    save_gz(output/'avaliacao_snapshot.json.gz',result)
    return result

def run_model(weight,records,model_name,environment,device,plan,policy,matcher,output,checkpoint=None,model_factory=None,predict_fn=None):
    weight=Path(weight)
    if core.sha256_file(weight)!=plan['weight_sha256'][model_name]:raise ValueError('Peso diferente do congelado.')
    validate_records(records,plan)
    if model_factory is None:
        from ultralytics import YOLO
        model_factory=YOLO
    model=model_factory(str(weight))
    try:
        names=model.names if isinstance(model.names,dict) else dict(enumerate(model.names))
        if names!=dict(enumerate(core.NAMES)):raise ValueError('Mapeamento de classes do peso incompatível.')
        cache,meta=obtain_cache(model,records,model_name,plan['weight_sha256'][model_name],environment,device,plan,
            Path(output)/'cache',checkpoint,predict_fn)
        result=evaluate(records,cache,model_name,plan,policy,matcher,output)
        result['cache_manifest']=meta
        return result
    finally:
        del model
