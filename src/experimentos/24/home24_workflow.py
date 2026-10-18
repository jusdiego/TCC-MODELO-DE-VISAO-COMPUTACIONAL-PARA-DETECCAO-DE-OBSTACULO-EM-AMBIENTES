"""Colab: entradas existentes, protocolo, referências, avaliação congelada e pacote."""
from __future__ import annotations
from pathlib import Path
from collections import deque
import gc,hashlib,json,shutil,time,zipfile
import home24_common as c
import home24_data as data
import home24_eval as ev

def sync_light(job,store):
    output=job['output']
    for p in sorted(output.rglob('*')):
        if not p.is_file() or any(x in p.parts for x in ['retomada','__pycache__']):continue
        if p.suffix in {'.ckpt','.tmp','.part','.zip'} or p.name.endswith('.transferindo'):continue
        if 'treino' in p.relative_to(output).parts:continue
        store.put(p,job['remote']+'/'+p.relative_to(output).as_posix())

def recover_outputs(job,store):
    # Somente esta execução, sem copiar outros experimentos nem seus bancos de dados.
    source=store.path(job['remote']);output=job['output']
    if not source.is_dir():return
    for p in source.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(source)
        if any(x in rel.parts for x in ['retomada','__pycache__','treino','codigo']):continue
        if p.suffix in {'.ckpt','.zip','.part','.tmp'} or p.name.endswith(('.transferindo','.sha256.json')):continue
        target=output/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.is_file():store.fetch(job['remote']+'/'+rel.as_posix(),target)

def start(store,local,bundle_sha,spec=None):
    spec=dict(c.SPEC if spec is None else spec);local=Path(local);local.mkdir(parents=True,exist_ok=True)
    boot=local/'preparacao';boot.mkdir(exist_ok=True)
    archive=store.fetch('datasets/homeobjects_sunrgbd_validado.zip',local/'entradas/home_historico.zip',c.EXPECTED['home_sha'])
    job=data.prepare(archive,local/'base',boot)
    initial=c.download_weight(local/'fontes/rf-detr-small.pth')
    env=c.environment()
    identity={'version':c.VERSION,'spec':spec,'dataset_id':job['reference']['dataset_id'],
      'bundle_sha256':bundle_sha,'rfdetr_pretrained_sha256':c.sha(initial),
      'packages':{k:env['packages'][k] for k in ['rfdetr','ultralytics','pytorch-lightning','transformers','supervision']},
      'torch_version':env['packages']['torch'].split('+')[0]}
    context=c.digest(identity);output=local/'resultados'/context[:24];output.mkdir(parents=True,exist_ok=True)
    remote='resultados/home24_rfdetr_'+context[:24]
    job.update(context=context,output=output,remote=remote,initial=initial,spec=spec,local=local,environment=env)
    recover_outputs(job,store)
    source=Path(__file__).parent
    if c.sha(source/'src/politica_alertas.py')!=c.EXPECTED['policy_sha'] or c.sha(source/'src/validacao_alertas.py')!=c.EXPECTED['matcher_sha']:
        raise ValueError('A política V2 ou o associador histórico foram alterados.')
    shutil.copytree(boot,output,dirs_exist_ok=True)
    shutil.copytree(source,output/'codigo',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    c.save(output/'configuracao_24.json',{'context':context,**identity})
    c.save(output/'ambiente_24.json',env)
    protocol={'version':c.VERSION,'context':context,'frozen_before_training':True,'dataset_id':job['reference']['dataset_id'],
      'primary':'P/R micro, 12 classes, IoU 0.50, Home TEST histórico',
      'secondary':'P/R nas seis categorias V2 no mesmo limiar principal; alertas V2 separados',
      'historical_test_exposed':True,'test_history':'Consultado em comparações anteriores, incluindo 03/05d e fluxo 10; não é teste novo intocado.',
      'train_images':len(job['train']),'val_images':len(job['val']),'test_images':len(job['test_descriptors']),
      'spec':spec,'references':{'candidato10':{'sha256':c.EXPECTED['c10_sha'],'imgsz':640},'21_epoca21':{'sha256':c.EXPECTED['21_sha'],'imgsz':768}},
      'new_model_resolution':spec['resolution'],'test_opening':'Somente após peso RF e limiares das três configurações fixados em VAL.',
      'external':'NYU opcional automático; diagnóstico de transferência, não comparação independente garantida.',
      'budget':'140 min de treino incluindo validação/persistência, ajustado ao alvo de 180 min da sessão; corte na fronteira da época.',
      'inference_postprocessing':'YOLO nativo com NMS IoU 0.7; RF-DETR nativo sem NMS adicional; configurações completas, não ablação de arquitetura.',
      'annotation_rule':'Sem alteração automática de rótulos. Exclusões históricas e proteção de duplicação registradas.',
      'test_does_not_select_model':True}
    frozen=store.path(remote+'/protocolo_24.json')
    if frozen.is_file() and c.digest(c.read(frozen))!=c.digest(protocol):raise ValueError('Protocolo remoto mudou para a mesma identidade.')
    c.save(output/'protocolo_24.json',protocol);sync_light(job,store)
    print('Protocolo registrado antes do treino. Pasta:',store.path(remote),flush=True)
    return job

def fetch_references(job,store):
    local=job['local']/'referencias';local.mkdir(exist_ok=True);refs={}
    refs['candidato10']={'path':store.fetch(c.EXPECTED['c10_path'],local/'candidato10.pt',c.EXPECTED['c10_sha']),
      'sha256':c.EXPECTED['c10_sha'],'imgsz':640,'epoch':'histórico'}
    candidates=[c.EXPECTED['21_path'],'resultados/treino21_5b35ca0c28f97027e7d1f324/melhor_modelo.pt']
    ref=None
    for rel in candidates:
        if store.path(rel).is_file():
            try:ref=store.fetch(rel,local/'21_epoca21.pt',c.EXPECTED['21_sha']);break
            except ValueError:continue
    if ref is None:
        archive_rel='resultados/treino21_5b35ca0c28f97027e7d1f324/pacote_treinamento_21.zip'
        if store.path(archive_rel).is_file():
            p=store.fetch(archive_rel,local/'pacote21.zip')
            with c.open_zip(p) as z:ref=c.install_checked(z.read('melhor_modelo.pt'),local/'21_epoca21.pt',c.EXPECTED['21_sha'])
    if ref is None:raise FileNotFoundError('Não localizei o checkpoint 21/época 21 preservado no Drive. Caminho esperado: '+str(store.path(c.EXPECTED['21_path'])))
    refs['21_epoca21']={'path':ref,'sha256':c.EXPECTED['21_sha'],'imgsz':768,'epoch':21}
    job['references']=refs;c.save(job['output']/'referencias_24.json',refs);return refs

def reference_validation(job,store,device=0):
    from ultralytics import YOLO
    job.setdefault('gpu_start',time.monotonic());refs=fetch_references(job,store);results={}
    for name,ref in refs.items():
        print('Avaliando referência em Home VAL:',name,flush=True)
        model=YOLO(str(ref['path']));folder=job['output']/'validacao'/name
        identity={'model':name,'sha256':ref['sha256'],'imgsz':ref['imgsz'],'nms_iou':.7,'precision':'fp32'}
        cache=ev.obtain_cache(job['val'],folder,identity,lambda rows:ev.predict_yolo(model,rows,ref['imgsz'],device),'val',checkpoint=lambda:sync_light(job,store))
        primary,secondary=ev.development(job['val'],cache,folder);results[name]={'primary':primary,'secondary':secondary,'identity':identity}
        del model;gc.collect()
    job['reference_results']=results;sync_light(job,store);return results

def load_rf(weight,device='cuda',resolution=512):
    from rfdetr import RFDETRSmall
    model=RFDETRSmall(pretrain_weights=str(weight),num_classes=12,resolution=resolution,device=device)
    # O dataset YOLO usa índices contínuos 0..11; não aplicar o deslocamento do COCO 1..90.
    names=model.model.class_names
    if isinstance(names,dict):names=[names[i] for i in range(len(names))]
    if list(names)!=c.NAMES:raise ValueError('O checkpoint RF-DETR perdeu a ordem/nome das 12 classes: '+str(names))
    return model

def evaluate(job,store,run,device='cuda'):
    if not run.get('completed'):raise ValueError('Treino não concluído; TEST permanece protegido.')
    output=job['output'];best=c.read(output/'melhor_24.json');weight=output/best['weight']
    if best['context']!=job['context'] or c.sha(weight)!=best['weight_sha256']:raise ValueError('Peso RF-DETR alterado.')
    shutil.copyfile(weight,output/'modelo_rfdetr24.pth')
    refs=job.get('reference_results')
    if refs is None:
        refs={name:{'primary':c.read(output/'validacao'/name/'metricas_12_classes.json'),
                    'secondary':c.read(output/'validacao'/name/'metricas_6_alerta.json')} for name in job['references']}
    predict_batch=c.read(output/'escolhas_execucao_24.json')['batch_size'];job['rf_predict_batch']=predict_batch
    model=load_rf(weight,device,job['spec']['resolution']);identity={'model':'rfdetr_small24','sha256':best['weight_sha256'],
      'resolution':job['spec']['resolution'],'postprocessing':'native_no_extra_nms','precision':'fp32','predict_batch':predict_batch}
    folder=output/'validacao/rfdetr_small24'
    cache=ev.obtain_cache(job['val'],folder,identity,lambda rows:ev.predict_rf(model,rows,predict_batch),'val',checkpoint=lambda:sync_light(job,store))
    primary,secondary=ev.development(job['val'],cache,folder)
    decisions={name:{'weight_sha256':info['sha256'],'threshold':refs[name]['primary']['threshold'],'imgsz':info['imgsz']} for name,info in job['references'].items()}
    decisions['rfdetr_small24']={'weight_sha256':best['weight_sha256'],'threshold':primary['threshold'],'resolution':job['spec']['resolution'],'epoch':best['epoch']}
    decision={'frozen':True,'context':job['context'],'dataset_id':job['reference']['dataset_id'],'models':decisions,
      'class_names':c.NAMES,'secondary_ids':c.ALERT_IDS,'match_iou':.5,'decision_uses_test':False,
      'test_history':'Home TEST histórico já exposto no projeto; não é confirmação em reserva nova.'}
    dp=output/'decisao_antes_test_24.json'
    if dp.is_file() and c.digest(c.read(dp))!=c.digest(decision):raise ValueError('A decisão já congelada não admite outro peso ou confiança.')
    c.save(dp,decision);store.put(dp,job['remote']+'/'+dp.name)
    # A publicação bem-sucedida acima é obrigatória antes de abrir os pixels/rótulos TEST.
    rows=data.open_historical_test(job,decision,output);tests={};secondaries={};frozen_key=c.digest(decision)
    for name in list(job['references'])+['rfdetr_small24']:
        target=output/'test_historico'/name;entry=decisions[name]
        if name=='rfdetr_small24':
            predictor=lambda block:ev.predict_rf(model,block,predict_batch);ident=identity
        else:
            from ultralytics import YOLO
            ref_model=YOLO(str(job['references'][name]['path']))
            predictor=lambda block:ev.predict_yolo(ref_model,block,entry['imgsz'],0 if device!='cpu' else 'cpu')
            ident={'model':name,'sha256':entry['weight_sha256'],'imgsz':entry['imgsz'],'nms_iou':.7,'precision':'fp32'}
        tc=ev.obtain_cache(rows,target,{**ident,'frozen_key':frozen_key},predictor,'test',checkpoint=lambda:sync_light(job,store))
        tests[name]=ev.profile(rows,tc,entry['threshold']);secondaries[name]=ev.profile(rows,tc,entry['threshold'],c.ALERT_IDS)
        ev.save_profiles(target,tests[name],secondaries[name])
        if name!='rfdetr_small24':del ref_model;gc.collect()
    ci=ev.bootstrap(tests);ci6=ev.bootstrap(secondaries);c.save(output/'bootstrap_test_historico_24.json',{'primary':ci,'secondary':ci6})
    job['test_rows']=rows
    result={'context':job['context'],'completed':True,'run':run,'best':best,'decision_key':frozen_key,
      'validation':{**{n:r['primary'] for n,r in refs.items()},'rfdetr_small24':primary},
      'validation_secondary':{**{n:r['secondary'] for n,r in refs.items()},'rfdetr_small24':secondary},
      'test_historico':tests,'test_historico_secondary':secondaries,'confidence_intervals':{'primary':ci,'secondary':ci6},
      'target_met_primary':tests['rfdetr_small24']['target_met'],'target_met_secondary':secondaries['rfdetr_small24']['target_met'],
      'test_is_new_independent':False,'manual_labels_required':False,'existing_model_replaced':False}
    c.save(output/'resumo_experimento_24.json',result);sync_light(job,store)
    job['rf_model']=model;job['rf_identity']=identity;return result

def _external_nyu(job,store,result):
    """Avaliação opcional sem ajustar nada com resultados NYU."""
    from PIL import Image
    output=job['output'];note={'executed':False,'purpose':'Diagnóstico externo em seis classes comuns, sem ajuste de pesos/limiares.'}
    decision=store.path('ramo_nyu22_23_v1/treinos/'+c.EXPECTED['nyu23_context'][:24]+'/decisao_antes_teste_nyu23.json')
    if not decision.is_file():note['reason']='Decisão congelada do 23 não localizada; teste externo omitido automaticamente.'
    elif time.monotonic()-job.get('gpu_start',time.monotonic())>c.SPEC['session_minutes']*60-10*60:note['reason']='Tempo reservado para fechamento; diagnóstico externo omitido.'
    else:
        frozen=c.read(decision)
        if frozen.get('dataset_id')!=c.EXPECTED['nyu_id'] or frozen.get('context')!=c.EXPECTED['nyu23_context'] or frozen.get('decision_uses_test') is not False:raise ValueError('Decisão NYU não corresponde ao ramo conhecido.')
        queue=deque((p,0) for p in [store.path('ramo_nyu22_23_v1'),store.path('nyu22_local')] if p.is_dir());seen=0;candidates=[]
        while queue and seen<160:
            folder,depth=queue.popleft();seen+=1
            for fn in ['atual22.json','pronto22.json']:
                p=folder/fn
                if p.is_file():
                    try:
                        info=c.read(p)
                        if info.get('dataset_id')==c.EXPECTED['nyu_id']:
                            expected_id=c.digest({k:info[k] for k in ['plan_key','development','reserve']})
                            if expected_id!=info['dataset_id'] or info['reserve']['sha256']!=c.EXPECTED['nyu_reserve_sha256']:raise ValueError('Identidade da preparação NYU alterada.')
                            candidates.append((p,info))
                    except (OSError,ValueError):continue
            if depth<7:
                for p in sorted(folder.iterdir()):
                    if p.is_dir() and (any(x in p.name.lower() for x in ['nyu','prepar','enviar','resultado','experimento']) or len(p.name) in {24,64}):queue.append((p,depth+1))
        archive_source=None;info=None
        for marker,item in candidates:
            for p in [marker.parent/item['reserve']['name'],store.path('ramo_nyu22_23_v1/'+item['remote']+'/'+item['reserve']['name'])]:
                if p.is_file():archive_source=p;info=item;break
            if archive_source:break
        if archive_source is None:note['reason']='Pacote de reserva NYU não localizado nas pastas preparadas; diagnóstico omitido.'
        else:
            archive=job['local']/'entradas/reserva_nyu22.zip';shutil.copyfile(archive_source,archive)
            if c.sha(archive)!=info['reserve']['sha256']:raise ValueError('Reserva NYU não corresponde à preparação congelada.')
            root=job['local']/'nyu_externo';mapping=[0,1,2,3,9,8];rows=[];excluded=[]
            hrows=job['train']+job['val']+job.get('test_rows',[]);pixels={r['pixel_sha256'] for r in hrows};phashes=[r['phash'] for r in hrows]
            with c.open_zip(archive) as z:
                manifest=json.loads(z.read('manifesto_reserva.json'))
                if manifest['classes']!=['bed','sofa','chair','table','door','window'] or len(manifest['test'])!=654:raise ValueError('Reserva NYU tem classes ou tamanho diferentes.')
                for raw in manifest['test']:
                    image=c.install_checked(z.read(raw['image']),root/raw['image'],raw['image_sha256'])
                    if data.pixel_sha(image) in pixels or data.near(data.phash(image),phashes):
                        excluded.append(raw['id']);continue
                    with Image.open(image) as im:w,h=im.size
                    gt=[{'classe_id':mapping[g['class_id']],'classe':c.NAMES[mapping[g['class_id']]],'bbox':g['bbox']} for g in raw['gt']]
                    rows.append({'id':raw['id'],'role':'external','image':image,'name':image.name,'width':w,'height':h,
                       'sha256':raw['image_sha256'],'label_sha256':raw['label_sha256'],'gt':gt,'scene_group':raw['scene_group']})
            if rows:
                dest=output/'nyu_externo';threshold=result['validation']['rfdetr_small24']['threshold']
                cache=ev.obtain_cache(rows,dest,{**job['rf_identity'],'frozen_key':result['decision_key']},lambda block:ev.predict_rf(job['rf_model'],block,job['rf_predict_batch']),'external',checkpoint=lambda:sync_light(job,store))
                metrics=ev.profile(rows,cache,threshold,mapping);c.save(dest/'metricas_6_comuns.json',metrics);c.csv_save(dest/'classes.csv',metrics['classes'])
                note.update(executed=True,images=len(rows),excluded_overlap_ids=excluded,metrics=metrics,
                  interpretation='Seis classes comuns; caixas NYU visíveis; sem igualdade de benchmark com o 23 após exclusões; não comprova independência entre fontes.')
            else:note.update(reason='Todas as imagens foram sinalizadas por sobreposição Home.',excluded_overlap_ids=excluded)
    c.save(output/'avaliacao_externa_24.json',note);result['external_nyu']=note;c.save(output/'resumo_experimento_24.json',result)
    job.pop('rf_model',None);gc.collect();sync_light(job,store);return note

def external_nyu(job,store,result):
    try:return _external_nyu(job,store,result)
    except Exception as exc:
        # A etapa externa é diagnóstica; uma falha não transforma resultados principais completos em treino interrompido.
        note={'executed':False,'reason':'Diagnóstico NYU não concluído: '+type(exc).__name__+': '+str(exc)[:900],
          'primary_results_preserved':True,'test_not_used_for_tuning':True}
        c.save(job['output']/'avaliacao_externa_24.json',note);result['external_nyu']=note
        c.save(job['output']/'resumo_experimento_24.json',result);job.pop('rf_model',None);gc.collect();sync_light(job,store)
        print(note['reason'],flush=True);return note

def failure(job,error,stage,store=None):
    output=job.get('output',job['local']/'diagnostico');output.mkdir(parents=True,exist_ok=True)
    c.save(output/'falha_24.json',{'stage':stage,'type':type(error).__name__,'message':str(error)[:2500],
      'manual_form_required':False,'test_opening_not_assumed':True,'action':'Preserve o runtime ou execute Run All novamente; retomada usa o último slot completo.'})
    p=c.package(output,'pacote_diagnostico_24.zip')
    if store and job.get('remote'):
        try:store.put(p,job['remote']+'/'+p.name)
        except Exception:pass
    print('Diagnóstico disponível em:',p,flush=True);return p
