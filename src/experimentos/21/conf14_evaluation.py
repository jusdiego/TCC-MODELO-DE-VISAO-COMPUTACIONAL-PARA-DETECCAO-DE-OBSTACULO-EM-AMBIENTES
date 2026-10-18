"""Varredura de confiança: uma inferência por imagem, política V2 fixa."""
from __future__ import annotations
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import diag11_core as core
import conf14_data as data
from exp12_protocol import judge

THRESHOLDS=(.05,.10,.15,.20,.25,.30,.40)
PREDICT_ARGS={'conf':.001,'imgsz':640,'rect':True,'iou':.70,'max_det':300,'half':False,'batch':1}
VERSION='14-confianca-automatica-1.0'

def point_name(conf):return 'conf_'+f'{conf:.2f}'.replace('.','')


def cache_request(records,weight_sha,environment):
    if not records or len({r['id'] for r in records})!=len(records):raise ValueError('Cache exige IDs únicos.')
    if any(r['role'] not in {'home_val','val_nova'} for r in records):raise ValueError('Cache somente para validação.')
    signature=[{k:r[k] for k in ['id','sha256','label_sha256','width','height','known_mask','role']} for r in records]
    return {'versao':VERSION,'peso_sha256':weight_sha,'parametros':PREDICT_ARGS,
            'ambiente':environment,'coorte':records[0]['coorte'],'imagens':len(records),
            'entradas_sha256':hashlib.sha256(core.canonical(signature).encode()).hexdigest()}

def read_cache(path,records):
    expected={r['id']:r for r in records};result={}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);iid=row['id']
            if iid not in expected or iid in result:raise ValueError('Cache com ID desconhecido ou repetido.')
            r=expected[iid]
            if row['image_sha256']!=r['sha256'] or row['label_sha256']!=r['label_sha256']:
                raise ValueError('Cache pertence a outros bytes de imagem/label.')
            validate_predictions(row['predicoes'],r);result[iid]=row['predicoes']
    if result.keys()!=expected.keys():raise ValueError('Cache incompleto.')
    return result


def event(record,kind,g,p,overlap,pred_low,conf):
    row=core.record_event(record,'B_e03',kind,g,p,overlap,pred_low)
    row['limiar_confianca']=conf
    row['melhor_confianca_mesma_classe']=None
    if kind=='objeto_nao_detectado':
        flags=[v for v in row['indicios'].split('|') if v and v not in {'deteccao_abaixo_da_confianca_025','sobreposicao_com_previsao_de_outra_classe','omissao_sem_indicio_automatico'}]
        matrix=core.iou_matrix([g],pred_low)[0]
        same=[i for i,p0 in enumerate(pred_low) if p0['classe_id']==g['classe_id']]
        if same:
            best=max(same,key=lambda i:matrix[i]);row['melhor_confianca_mesma_classe']=pred_low[best]['confianca']
            if matrix[best]>=.5 and pred_low[best]['confianca']<conf:flags.append('deteccao_abaixo_do_limiar_avaliado')
        if any(matrix[i]>=.5 and p0['classe_id']!=g['classe_id'] and p0['confianca']>=conf for i,p0 in enumerate(pred_low)):
            flags.append('sobreposicao_com_previsao_de_outra_classe')
        row['indicios']='|'.join(flags or ['omissao_sem_indicio_automatico'])
    return row

def analyse(record,pred_low,conf,policy,matcher):
    if record['role'] not in {'home_val','val_nova'} or conf not in THRESHOLDS:
        raise ValueError('Papel ou limiar fora do protocolo.')
    known=int(record['known_mask'])
    gt=core.with_policy([g for g in record['gt'] if known&(1<<g['classe_id'])],record['width'],record['height'],policy)
    low=[p for p in pred_low if known&(1<<p['classe_id'])]
    all_pred=core.with_policy([p for p in pred_low if p['confianca']>=conf],record['width'],record['height'],policy)
    pred=[p for p in all_pred if known&(1<<p['classe_id'])]
    pairs,missing,extras=matcher(gt,pred)
    events=[event(record,'correspondencia',gt[gi],pred[pi],float(iou),low,conf) for gi,pi,iou in pairs]
    events += [event(record,'objeto_nao_detectado',gt[gi],None,None,low,conf) for gi in missing]
    events += [event(record,'previsao_nao_associada',None,pred[pi],None,low,conf) for pi in extras]
    messages=[policy.gerar_mensagem(p,'medio') for p in all_pred if core.alert_on(p)]
    if any(not isinstance(s,str) or not s.strip() for s in messages):raise RuntimeError('Alerta sem mensagem.')
    im={'coorte':record['coorte'],'imagem':record['name'],'id':record['id'],'modelo':'B_e03','limiar_confianca':conf,
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
    return im,events

def summarise(images,events,conf):
    if not images:raise ValueError('Validação vazia.')
    total=lambda k:sum(r[k] for r in images)
    complete=[r for r in images if r['cobertura_completa_alertas']]
    summary={'modelo':'B_e03','coorte':images[0]['coorte'],'imagens':len(images),'conf_alertas':conf,
        'alertas_por_objeto':core.alert_metrics(total('tp_alerta'),total('fn_alerta'),total('fp_alerta')),
        'alta_ref':total('alta_ref'),'alta_sem_alerta':total('alta_sem_alerta'),
        'taxa_alta_sem_alerta':core.ratio(total('alta_sem_alerta'),total('alta_ref')),
        'detector_no_limiar':core.alert_metrics(total('tp_detector'),total('fn_detector'),total('fp_detector')),
        'gt_ignorados_sem_cobertura':total('gt_ignorados'),'predicoes_ignoradas_conf':total('predicoes_ignoradas_conf'),
        'imagens_com_cobertura_completa_alertas':len(complete),
        'alertas_por_imagem_cobertura_completa':core.alert_metrics(
            sum(r['ha_alerta_ref'] and r['ha_alerta_pred'] for r in complete),
            sum(r['ha_alerta_ref'] and not r['ha_alerta_pred'] for r in complete),
            sum(not r['ha_alerta_ref'] and r['ha_alerta_pred'] for r in complete)),
        'referencia':'Concordância com política V2 nas caixas; não valida risco físico.',
        'nao_e_openimages_challenge':True,'ap_recalculada':False}
    classes=[]
    for cid,name in enumerate(core.NAMES):
        part=[e for e in events if e['classe_id']==cid];n=sum(e['bbox_ref']!='' for e in part)
        classes.append({'classe_id':cid,'classe':name,'gt_avaliaveis':n,'suporte_reduzido':n<30,'conf_alertas':conf,
            **core.alert_metrics(sum(e['ha_alerta_ref'] and e['ha_alerta_pred'] for e in part),
                                 sum(e['ha_alerta_ref'] and not e['ha_alerta_pred'] for e in part),
                                 sum(not e['ha_alerta_ref'] and e['ha_alerta_pred'] for e in part)),
            'alta_ref':sum(e['prioridade_ref']=='alto' for e in part),
            'alta_sem_alerta':sum(e['prioridade_ref']=='alto' and not e['ha_alerta_pred'] for e in part)})
    return summary,classes

def evaluate_cache(records,cache,conf,policy,matcher,output=None):
    if cache.keys()!={r['id'] for r in records}:raise ValueError('Cache não corresponde aos IDs da validação.')
    images=[];events=[]
    for r in records:
        im,ev=analyse(r,cache[r['id']],conf,policy,matcher);images.append(im);events.extend(ev)
    summary,classes=summarise(images,events,conf)
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        core.save_json(output/'metricas.json',summary)
        core.save_csv(output/'metricas_por_classe.csv',classes)
        core.save_csv(output/'alertas_por_imagem.csv',images)
        core.save_csv(output/'avaliacao_objetos.csv',events)
    return {'summary':summary,'classes':classes,'images':images,'events':events}

def parse_current_events(events):
    result=[]
    for r in events:
        v=dict(r)
        for key in ['bbox_ref','bbox_pred']:v[key]=json.loads(v[key]) if v[key] else None
        result.append(v)
    return result

def check_reproduction(result,original,cohort):
    old={r['id']:r for r in original['images']};new={r['id']:r for r in result['images']}
    if old.keys()!=new.keys():raise ValueError('Reprodução exige os mesmos IDs.')
    rows=[]
    fields=['tp_alerta','fn_alerta','fp_alerta','alta_ref','alta_sem_alerta','tp_detector','fn_detector','fp_detector','gt_avaliaveis','gt_ignorados','predicoes_ignoradas_conf_025']
    for iid in sorted(old):
        for key in fields:
            key_new='predicoes_ignoradas_conf' if key=='predicoes_ignoradas_conf_025' else key
            a,b=old[iid][key],new[iid][key_new]
            rows.append({'coorte':cohort,'id':iid,'metrica':key,'notebook12':a,'notebook14':b,'igual':a==b})
    old_gt=data.gt_map(original['events']);new_gt=data.gt_map(parse_current_events(result['events']))
    if old_gt.keys()!=new_gt.keys():raise ValueError('Referências GT mudaram na reprodução.')
    object_rows=[]
    for key,a in old_gt.items():
        b=new_gt[key]
        fields_gt=['evento','prioridade_ref','prioridade_pred','ha_alerta_ref','ha_alerta_pred']
        diffs=[field for field in fields_gt if a[field]!=b[field]]
        object_rows.append({'coorte':cohort,'id':key[0],'classe':a['classe'],'bbox_ref':json.dumps(key[2]),
                            'campos_diferentes':'|'.join(diffs),'igual':not diffs})
    return {'coorte':cohort,'checagens_imagem':len(rows),'divergencias_imagem':sum(not r['igual'] for r in rows),
            'objetos_GT':len(object_rows),'divergencias_objetos':sum(not r['igual'] for r in object_rows),
            'reproduziu':all(r['igual'] for r in rows+object_rows)},rows,object_rows

def select_threshold(results,rule,reproduction_ok):
    assessed=[]
    for conf in THRESHOLDS:
        home=results['home_val_historica'][conf]['summary'];ext=results['openimages_val_nova'][conf]['summary']
        supported=all(s['alertas_por_objeto'][k] is not None and math.isfinite(s['alertas_por_objeto'][k])
                      for s in [home,ext] for k in ['precision','recall','f1'])
        if supported:
            checks={**judge(home,ext,rule),'metricas_com_suporte':True}
        else:
            # Precisão sem nenhum alerta é indefinida; não vira zero ou aprovação.
            checks={k:None for k in ['home_precision_superior','home_recall_preservado','home_altas_preservadas',
                                    'externa_precision_preservada','externa_recall_preservado','externa_altas_preservadas']}
            checks.update(elegivel=False,retencao_externa=False,metricas_com_suporte=False)
        assessed.append({'conf':conf,'home':home,'external':ext,'checks':checks,
                         'comparavel_historico':reproduction_ok,'elegivel':checks['elegivel'] and reproduction_ok})
    def ranking(r):
        h=r['home']['alertas_por_objeto'];e=r['external']['alertas_por_objeto']
        return (-h['precision'],-h['recall'],r['home']['alta_sem_alerta'],-e['f1'],abs(r['conf']-.25),-r['conf'])
    good=[r for r in assessed if r['elegivel']]
    chosen=min(good,key=ranking) if good else None
    status='limiar_candidato_em_validacao' if chosen else 'nenhum_limiar_elegivel'
    if not reproduction_ok:status='reproducao_divergente_sem_sugestao_de_limiar'
    return {'status':status,'limiar_sugerido':chosen['conf'] if chosen else None,
            'comparabilidade_confirmada':bool(reproduction_ok),'assessed':assessed,
            'limiar_galeria':chosen['conf'] if chosen else .10,
            'criterio_galeria_sem_elegivel':'0,10 fixado antes da inferência; somente diagnóstico.',
            'modelo_em_uso_continua_baseline':True,'substituicao_automatica':False}

def effects(results,bundle,cohort,cache):
    old=data.gt_map(bundle['evaluations']['candidato_10'][cohort]['events'])
    anchor=data.gt_map(parse_current_events(results[cohort][.25]['events']))
    lost={k for k in anchor if old[k]['ha_alerta_ref'] and old[k]['ha_alerta_pred'] and not anchor[k]['ha_alerta_pred']}
    changed=[];losses=[];priority=[]
    for key in sorted(lost):
        a,b=old[key],anchor[key]
        if b['evento']=='correspondencia':
            priority.append({'coorte':cohort,'id':key[0],'classe':b['classe'],'bbox_ref':json.dumps(key[2]),
                'prioridade_candidato10':a['prioridade_pred'],'prioridade_B_025':b['prioridade_pred'],
                'confianca_candidato10':a['confianca'],'confianca_B_025':b['confianca'],
                'observacao':'Objeto ainda associado; alerta pode mudar pela geometria/prioridade.'})
    for conf in THRESHOLDS:
        current=data.gt_map(parse_current_events(results[cohort][conf]['events']))
        for key,b in current.items():
            a=anchor[key]
            if a['ha_alerta_ref'] and a['ha_alerta_pred']!=b['ha_alerta_pred']:
                changed.append({'coorte':cohort,'conf':conf,'id':key[0],'classe':b['classe'],'bbox_ref':json.dumps(key[2]),
                    'alerta_B_025':a['ha_alerta_pred'],'alerta_no_limiar':b['ha_alerta_pred'],
                    'prioridade_ref':b['prioridade_ref'],'confianca_associada':b['confianca']})
            if key not in lost:continue
            same=[p for p in cache[key[0]] if p['classe_id']==key[1]]
            ious=core.iou_matrix([{'bbox':list(key[2])}],same)[0]
            best=max(range(len(same)),key=lambda i:ious[i]) if same else None
            losses.append({'coorte':cohort,'conf':conf,'id':key[0],'classe':b['classe'],'bbox_ref':json.dumps(key[2]),
                'prioridade_ref':b['prioridade_ref'],'recuperado':b['ha_alerta_pred'],
                'evento_B_025':a['evento'],'evento_no_limiar':b['evento'],
                'melhor_iou_cache_mesma_classe':float(ious[best]) if best is not None else None,
                'confianca_dessa_proposta':same[best]['confianca'] if best is not None else None,
                'limite':'Melhor IoU não garante associação nem alerta; coluna recuperado usa avaliação completa.'})
    return changed,losses,priority


def validate_predictions(pred,record):
    """Valida estrutura; preserva coordenadas numéricas como no avaliador 11/12.

    Área zero e limites da imagem são diagnósticos, não filtros do cache.
    Nenhuma proposta é removida, recortada ou corrigida aqui.
    """
    totals={'predicoes':len(pred),'area_zero':0,'fora_da_imagem':0,
            'area_zero_na_grade':0,'fora_da_imagem_na_grade':0,'amostras':[]}
    for index,p in enumerate(pred):
        reason=None
        if not isinstance(p,dict):
            raise ValueError(f"Previsão não é um dicionário: {record['id']}; índice={index}; valor={p!r}")
        cid=p.get('classe_id');box=p.get('bbox');score=p.get('confianca')
        if type(cid)!=int or not 0<=cid<len(core.NAMES):reason='classe_id inválida'
        elif p.get('classe')!=core.NAMES[cid]:reason='nome da classe incompatível'
        elif type(score) not in (int,float) or not math.isfinite(score) or not .001-1e-8<=score<=1:
            reason='confiança não finita ou fora da faixa de captura'
        elif not isinstance(box,(list,tuple)) or len(box)!=4:reason='bbox deve ter quatro coordenadas'
        elif not all(type(v) in (int,float) and math.isfinite(v) for v in box):reason='coordenada não numérica ou não finita'
        elif box[2]<box[0] or box[3]<box[1]:reason='coordenadas xyxy invertidas'
        if reason:
            raise ValueError(f"Previsão incompatível com o cache: {record['id']}; motivo={reason}; "
                f"índice={index}; classe_id={cid!r}; confiança={score!r}; bbox={box!r}; "
                f"dimensões=(w={record['width']}, h={record['height']})")
        zero=box[2]==box[0] or box[3]==box[1]
        outside=(min(box)<-.01 or box[2]>record['width']+.01 or box[3]>record['height']+.01)
        totals['area_zero']+=int(zero);totals['fora_da_imagem']+=int(outside)
        totals['area_zero_na_grade']+=int(zero and score>=min(THRESHOLDS))
        totals['fora_da_imagem_na_grade']+=int(outside and score>=min(THRESHOLDS))
        if (zero or outside) and len(totals['amostras'])<5:
            totals['amostras'].append({'indice':index,'classe':p['classe'],'confianca':score,
                'bbox':list(box),'area_zero':zero,'fora_da_imagem':outside,
                'width':record['width'],'height':record['height']})
    return totals

def save_geometry_diagnostics(directory,items):
    totals={key:sum(item['diagnostico'][key] for item in items) for key in
            ['predicoes','area_zero','fora_da_imagem','area_zero_na_grade','fora_da_imagem_na_grade']}
    unusual=[item for item in items if item['diagnostico']['area_zero'] or item['diagnostico']['fora_da_imagem']]
    report={'correcao':'14-cache-validacao-1.0.1','imagens':len(items),
        'imagens_com_geometria_incomum':len(unusual),'totais':totals,
        'propostas_removidas':0,'coordenadas_alteradas':False,
        'observacao':'Geometria incomum preservada no cache e no protocolo; não constitui objeto aprovado. Reprodução em 0,25 continua obrigatória.',
        'casos':unusual}
    core.save_json(Path(directory)/'diagnostico_geometria_previsoes.json',report)
    return {k:v for k,v in report.items() if k!='casos'}

def obtain_cache(model,records,directory,weight_sha,environment,device,predict_fn=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'predicoes_conf_0p001.jsonl.gz';meta_path=directory/'cache_manifesto.json'
    request=cache_request(records,weight_sha,environment)
    if path.is_file() and meta_path.is_file():
        meta=json.loads(meta_path.read_text())
        if meta['solicitacao']!=request or meta['cache_sha256']!=core.sha256_file(path):
            raise RuntimeError('Cache existente não corresponde aos pesos/entradas/ambiente. Não será reutilizado.')
        cached=read_cache(path,records)
        diagnostics=[{'id':r['id'],'diagnostico':validate_predictions(cached[r['id']],r)} for r in records]
        geometry=save_geometry_diagnostics(directory,diagnostics)
        print('Cache conferido e reutilizado:',records[0]['coorte'],flush=True)
        return path,{**meta,'geometria_resumo':geometry,'reutilizado_nesta_execucao':True}
    predict=predict_fn or core.model_predictions
    tmp=path.with_name(path.name+'.tmp')
    saturated=0;diagnostics=[]
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=4) as f:
        for i,r in enumerate(records,1):
            if r['role'] not in {'home_val','val_nova'}:raise ValueError('Papel proibido na inferência.')
            pred=predict(model,r,device)
            try:diagnostic=validate_predictions(pred,r)
            except ValueError as exc:
                core.save_json(directory/'previsao_rejeitada.json',{'id':r['id'],'mensagem':str(exc),
                    'width':r['width'],'height':r['height'],'propostas':pred})
                raise
            diagnostics.append({'id':r['id'],'diagnostico':diagnostic})
            if len(pred)==PREDICT_ARGS['max_det']:saturated+=1
            f.write(json.dumps({'id':r['id'],'imagem':r['name'],'image_sha256':r['sha256'],
                                'label_sha256':r['label_sha256'],'predicoes':pred},ensure_ascii=False)+'\n')
            if i%100==0 or i==len(records):print(f"Checkpoint / {r['coorte']}: {i}/{len(records)}",flush=True)
    tmp.replace(path)
    geometry=save_geometry_diagnostics(directory,diagnostics)
    meta={'solicitacao':request,'cache_sha256':core.sha256_file(path),'concluido':True,
          'validacao_previsoes':'14-cache-validacao-1.0.1','geometria_resumo':geometry,
          'imagens_atingiram_max_det':saturated,'limite':'Propostas pós-NMS em confiança 0,001; max_det=300, sem acesso aos logits brutos.'}
    core.save_json(meta_path,meta)
    read_cache(path,records)
    print('Diagnóstico geométrico do cache:',geometry['totais'],flush=True)
    return path,{**meta,'reutilizado_nesta_execucao':False}
