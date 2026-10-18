"""Avaliação fixa em 0,25 com avaliador autenticado e isolamento do RNG."""
from __future__ import annotations
import gzip,json
from pathlib import Path
from ultralytics import YOLO
import diag11_core as core
import conf14_evaluation as e14
from exp12_training import isolated_evaluation
import mix18_protocol as p16

def evaluate(weight,checkpoint_id,cohorts,output,policy,matcher,environment,device):
    output=Path(output);sha=core.sha256_file(weight);results={}
    if set(cohorts)!={'home_val_historica','openimages_val_nova'}:raise ValueError('Coortes inesperadas.')
    with isolated_evaluation():
        model=YOLO(str(weight))
        names=model.names if isinstance(model.names,dict) else dict(enumerate(model.names))
        if names!=dict(enumerate(core.NAMES)):raise ValueError('Classes do modelo incompatíveis.')
        for cohort,records in cohorts.items():
            target=output/cohort
            cache,meta=e14.obtain_cache(model,records,target/'cache',sha,environment,device)
            preds=e14.read_cache(cache,records)
            result=e14.evaluate_cache(records,preds,.25,policy,matcher,target)
            result['summary']['modelo']=checkpoint_id
            for key in ['images','events']:
                for row in result[key]:row['modelo']=checkpoint_id
            core.save_json(target/'metricas.json',result['summary'])
            core.save_csv(target/'alertas_por_imagem.csv',result['images']);core.save_csv(target/'avaliacao_objetos.csv',result['events'])
            # Cache compacto preserva todas as previsões que participaram da associação em 0,25.
            compact=target/'predicoes_conf_025.jsonl.gz'
            with gzip.open(compact,'wt',encoding='utf-8',compresslevel=4) as f:
                for r in records:f.write(json.dumps({'id':r['id'],'image_sha256':r['sha256'],'label_sha256':r['label_sha256'],
                    'predicoes':[p for p in preds[r['id']] if p['confianca']>=.25]},ensure_ascii=False)+'\n')
            result['cache_manifest']=meta;result['compact_cache_sha256']=core.sha256_file(compact)
            results[cohort]=result
        del model
    p16.save_gz(output/'avaliacao_snapshot.json.gz',core.json_clean(results))
    return results

def entry(checkpoint_id,arm,epoch,updates,relative_weight,weight_sha,evaluated):
    return {'id':checkpoint_id,'arm':arm,'epoch':epoch,'updates':updates,
            'weight_relative':relative_weight,'weight_sha256':weight_sha,
            'home':evaluated['home_val_historica']['summary'],
            'external':evaluated['openimages_val_nova']['summary']}
