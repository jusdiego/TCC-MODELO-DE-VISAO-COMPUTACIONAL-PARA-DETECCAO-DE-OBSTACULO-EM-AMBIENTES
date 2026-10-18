"""Orquestra a preparação 22 e a sessão 23; não acessa pastas do experimento 21."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import platform,shutil,time
import nyu22_common as c
import nyu22_data as data

def preparation(storage,local,output,bundle,spec=c.DATA_SPEC,sources=c.SOURCES):
    local=Path(local);output=Path(output);local.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    plan={'spec':spec,'sources':{k:sources[k] for k in ['mat','split']},'code_bundle':bundle}
    plan_key=c.digest(plan);remote='preparado/'+plan_key[:24];ready=storage.path(remote+'/pronto22.json')
    if ready.is_file():
        info=c.read(ready)
        if info['plan_key']!=plan_key:raise ValueError('Preparação remota pertence a outro plano.')
        for key in ['development','reserve']:
            if not storage.path(remote+'/'+info[key]['name']).is_file():raise ValueError('A preparação anterior ficou sem um dos pacotes de dados.')
        for name in ['preparacao_nyu22.json','classes_desenvolvimento.csv','exemplos_desenvolvimento.jpg','pacote_preparacao_nyu22.zip']:
            if storage.path(remote+'/'+name).is_file():storage.pull(remote+'/'+name,output/name)
        storage.put(ready,'atual22.json');return info
    c.save(output/'plano_preparacao_nyu22.json',plan)
    print('Baixando/conferindo o arquivo oficial anotado (~2,8 GiB). Não é o conjunto RAW de 428 GB.',flush=True)
    mat=storage.source('mat',local/'fontes',sources);split=storage.source('split',local/'fontes',sources)
    report,dev,reserve=data.prepare(mat,split,local/'base',output,spec)
    info={'version':spec['version'],'plan_key':plan_key,'remote':remote,'classes':c.NAMES,
        'development':{'name':dev.name,'sha256':c.sha(dev),'bytes':dev.stat().st_size},
        'reserve':{'name':reserve.name,'sha256':c.sha(reserve),'bytes':reserve.stat().st_size},
        'sources':sources,'report':report,'test_used_for_selection':False}
    info['dataset_id']=c.digest({'plan_key':plan_key,'development':info['development'],'reserve':info['reserve']})
    c.save(output/'pronto22.json',info)
    # A preparação só é publicada como atual depois dos dois pacotes completos.
    storage.put(dev,remote+'/'+dev.name);storage.put(reserve,remote+'/'+reserve.name)
    for name in ['preparacao_nyu22.json','classes_desenvolvimento.csv','exemplos_desenvolvimento.jpg','plano_preparacao_nyu22.json']:
        storage.put(output/name,remote+'/'+name)
    archive=c.package(output,'pacote_preparacao_nyu22.zip');storage.put(archive,remote+'/'+archive.name)
    storage.put(output/'pronto22.json',remote+'/pronto22.json');storage.put(output/'pronto22.json','atual22.json')
    return info

def session(storage,local,bundle,spec=c.TRAIN_SPEC,device=0):
    import torch,ultralytics,numpy,PIL
    import nyu23_training as tr
    if ultralytics.__version__!='8.4.150':raise RuntimeError('A versão importada difere da fixada. Reinicie a sessão e use Run All.')
    local=Path(local);info=c.read(storage.path('atual22.json'))
    if info['classes']!=c.NAMES:raise ValueError('Execute a preparação 22 desta linha antes do treinamento 23.')
    base=local/'base'/info['dataset_id'][:24];base.mkdir(parents=True,exist_ok=True)
    archive=storage.pull(info['remote']+'/'+info['development']['name'],base/'desenvolvimento.zip',info['development']['sha256'])
    train,val,yaml_file,manifest=data.load_development(archive,base)
    identity={'version':spec['version'],'code_bundle':bundle,'dataset_id':info['dataset_id'],'spec':spec,'ultralytics':'8.4.150','pretrained_sha256':c.SOURCES['weight']['sha256']}
    context=c.digest(identity);output=local/'resultados'/context[:24];output.mkdir(parents=True,exist_ok=True);remote='treinos/'+context[:24]
    if tr.local_resume(output,context) is None:tr.restore(storage,output,remote,context)
    c.save(output/'configuracao_nyu23.json',{'context':context,**identity});c.save(output/'base_nyu22.json',info)
    shutil.copy2(base/'manifesto_desenvolvimento.json',output/'manifesto_desenvolvimento_referencia.json')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    environment={'python':platform.python_version(),'torch':torch.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,
        'ultralytics':ultralytics.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda':torch.version.cuda,'amp':False,'quantize':32}
    c.save(output/'sessoes'/(stamp+'.json'),environment)
    initial=storage.source('weight',local/'fontes')
    return {'context':context,'output':output,'remote':remote,'train':train,'val':val,'yaml':yaml_file,'manifest':manifest,'info':info,'base':base,'initial':initial,'environment':environment}

def evaluate_session(storage,job,run,device,spec=c.TRAIN_SPEC,backup=None):
    from ultralytics import YOLO
    import nyu23_eval as ev
    output=job['output'];best=c.read(output/'melhor_nyu23.json');weight=output/best['weight']
    if best['context']!=job['context'] or c.sha(weight)!=best['weight_sha256']:raise ValueError('Melhor peso alterado ou de outra execução.')
    shutil.copy2(weight,output/'melhor_modelo.pt');model=YOLO(str(weight))
    decision_path=output/'decisao_antes_teste_nyu23.json';frozen=c.read(decision_path) if decision_path.is_file() else None
    if frozen and (frozen['context']!=job['context'] or frozen['weight_sha256']!=best['weight_sha256']):raise ValueError('A decisão já congelada aponta para outro modelo.')
    selected,profiles=ev.evaluate_development(model,job['val'],device,spec,output/'validacao',best['weight_sha256'],frozen['threshold'] if frozen else None)
    test=None
    if run['completed']:
        if frozen is None:
            frozen={'context':job['context'],'weight_sha256':best['weight_sha256'],'threshold':selected['threshold'],'epochs':run['epochs'],
                'classes':c.NAMES,'dataset_id':job['info']['dataset_id'],'inference':{k:spec[k] for k in ['imgsz','inference_conf','nms_iou','match_iou','max_det']},
                'decision_uses_test':False,'selection':'Somente VAL; peso e limiar fixados antes de consultar previsões de TEST.'}
            c.save(decision_path,frozen)
        # Falha neste envio interrompe antes da primeira previsão na reserva.
        storage.put(decision_path,job['remote']+'/'+decision_path.name)
        test_dir=output/'teste_final';cached=test_dir/'metricas.json';frozen_key=c.digest(frozen)
        if cached.is_file():
            test=c.read(cached)
            if test['frozen_key']!=frozen_key or not test['complete']:raise ValueError('Resultado de TEST não corresponde à decisão congelada.')
            print('Teste final já concluído: reutilizando o resultado congelado.',flush=True)
        else:
            info=job['info'];archive=storage.pull(info['remote']+'/'+info['reserve']['name'],job['base']/'reserva.zip',info['reserve']['sha256'])
            rows=data.load_reserve(archive,job['base'],job['manifest'])
            c.save(output/'abertura_teste_nyu23.json',{'frozen_key':frozen_key,'images':len(rows),'weights_or_thresholds_changed':False})
            predictions=ev.obtain_cache(model,rows,device,spec,test_dir/'cache',best['weight_sha256'],'test',checkpoint=backup)
            test=ev.profile(rows,predictions,frozen['threshold'],spec);test['confidence_intervals']=ev.scene_bootstrap(test['image_rows'],spec)
            test_yaml=data.write_yaml(job['train'],rows,job['base'],'test_data.yaml')
            native=model.val(data=str(test_yaml),imgsz=spec['imgsz'],batch=spec['predict_batch'],device=device,
                conf=.001,iou=spec['nms_iou'],max_det=spec['max_det'],quantize=32,workers=spec['workers'],
                plots=False,save_json=False,verbose=False,project=str(output),name='ap_teste',exist_ok=True)
            test.update(native_ap={'map50':float(native.box.map50),'map50_95':float(native.box.map)},frozen_key=frozen_key,complete=True)
            c.save(cached,test);c.csv_save(test_dir/'classes.csv',test['classes']);c.csv_save(test_dir/'imagens.csv',test['image_rows'])
    del model
    return best,selected,test

def failure(output,error,stage,storage=None,remote=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    c.save(output/'resumo_nyu23.json',{'completed':False,'status':'interrompido','stage':stage,'error':{'type':type(error).__name__,'message':str(error)[:1600]},'manual_form_required':False})
    archive=c.package(output,'pacote_diagnostico_nyu.zip')
    if storage and remote:
        try:storage.put(archive,remote+'/'+archive.name)
        except Exception:pass
    return archive
