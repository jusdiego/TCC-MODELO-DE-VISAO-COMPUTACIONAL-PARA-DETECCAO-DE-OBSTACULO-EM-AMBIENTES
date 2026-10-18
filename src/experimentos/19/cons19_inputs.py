"""Resultados autenticados e previsões já capturadas; sem Torch ou inferência."""
from __future__ import annotations
from collections import Counter
import csv,gzip,hashlib,io,json,math,zipfile
from pathlib import Path
import diag11_core as core
import conf14_data as data
import conf14_evaluation as evaluator
import cons19_protocol as protocol

def rows(raw):return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
def checked_zip(path,sha):
    if core.sha256_file(path)!=sha:raise ValueError('Pacote diferente do resultado analisado: '+Path(path).name)
    z=zipfile.ZipFile(path)
    try:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip():raise ValueError('Pacote com nomes repetidos ou CRC inválido.')
        for n in z.namelist():core.safe_member(n)
        return z
    except Exception:z.close();raise

def load(packages,expected,output):
    output=Path(output);sources={};all_entries=[];references={};caches={}
    for number in [16,17,18]:
        with checked_zip(packages[number],expected['pacote_'+str(number)]) as z:
            member='resumo_calibracao_17.json' if number==17 else f'resumo_experimento_{number}.json'
            summary=json.loads(z.read(member))
            if not summary['concluido']:raise ValueError('Resultado anterior incompleto.')
            sources[number]={'summary':summary,'sha256':expected['pacote_'+str(number)]}
            if number==17:
                if summary['eligible_calibration']!=0 or summary['distinct_configurations_evaluated_verification']!=0:
                    raise ValueError('Resultado 17 diferente da decisão que encerrou a busca de limiares.')
                records=json.loads(gzip.decompress(z.read('registros_validacao.json.gz')))
                if len(records)!=1456 or len({r['id'] for r in records})!=1456:raise ValueError('Registros de validação incompatíveis.')
                if Counter(r['role'] for r in records)!={'home_val':456,'val_nova':1000}:raise ValueError('Papel protegido entrou nos registros.')
                sources[number]['records']=records
            else:
                if any(summary[k] for k in ['reserva_futura_inferida','test_home_inferido','test_oi20k_inferido','modelo_substituido']):
                    raise ValueError('Resultado anterior usou conjunto protegido ou substituiu modelo.')
                entries=json.loads(z.read('checkpoints_avaliados.json'))
                finals=[e for e in entries if e['arm']!='inicial' and e['epoch']==3]
                if len(finals)!=2 or summary['checkpoint_final_elegivel'] is not None:raise ValueError('Finais anteriores incompatíveis.')
                for entry in entries:
                    if entry['arm']!='inicial':all_entries.append({'experiment':number,**entry})
                if number==18:
                    cfg=json.loads(z.read('configuracao_experimento_18.json'))
                    if cfg['context_key']!=expected['contexto_18']:raise ValueError('Contexto 18 diferente.')
                    raw_rule=z.read('regra_selecao_congelada_12.json')
                    if hashlib.sha256(raw_rule).hexdigest()!=expected['regra_12_sha256']:raise ValueError('Regra histórica diferente.')
                    (output/'regra_selecao_congelada_12.json').write_bytes(raw_rule)
                    rule=json.loads(raw_rule)
                    for model in protocol.MODELS:
                        references[model]={};caches[model]={}
                        for cohort in protocol.COHORTS:
                            prefix=f'referencias/{model}/{cohort}/'
                            meta=json.loads(z.read(prefix+'cache/cache_manifesto.json'))
                            request=meta['solicitacao'];weight=expected['baseline' if model=='baseline' else 'candidato']
                            if not meta['concluido'] or request['peso_sha256']!=weight or request['coorte']!=cohort:
                                raise ValueError('Cache de outra referência.')
                            raw=z.read(prefix+'predicoes_conf_025.jsonl.gz')
                            values=[json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l]
                            references[model][cohort]={'summary':json.loads(z.read(prefix+'metricas.json')),
                                'images':rows(z.read(prefix+'alertas_por_imagem.csv')),
                                'events':data.parse_events(z.read(prefix+'avaliacao_objetos.csv')),
                                'classes':rows(z.read(prefix+'metricas_por_classe.csv')),'request':request}
                            caches[model][cohort]=values
                            target=output/'caches_reutilizados'/model/(cohort+'.jsonl.gz')
                            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    records=sources[17]['records'];by_id={r['id']:r for r in records};cohorts={};predictions={m:{} for m in protocol.MODELS}
    for cohort in protocol.COHORTS:
        ordered=[v['id'] for v in caches['baseline'][cohort]]
        wanted={r['id'] for r in records if r['coorte']==cohort}
        if len(ordered)!=len(wanted) or set(ordered)!=wanted:raise ValueError('Cache e GT usam IDs diferentes.')
        cohorts[cohort]=[by_id[iid] for iid in ordered]
        for model in protocol.MODELS:
            values=caches[model][cohort]
            if [v['id'] for v in values]!=ordered:raise ValueError('Ordem dos caches diverge entre modelos.')
            sig=[{k:r[k] for k in ['id','sha256','label_sha256','width','height','known_mask','role']} for r in cohorts[cohort]]
            if hashlib.sha256(core.canonical(sig).encode()).hexdigest()!=references[model][cohort]['request']['entradas_sha256']:
                raise ValueError('Bytes, dimensões ou cobertura diferem da referência.')
            predictions[model][cohort]={}
            for item in values:
                r=by_id[item['id']]
                if item['image_sha256']!=r['sha256'] or item['label_sha256']!=r['label_sha256']:raise ValueError('Cache de outros bytes.')
                evaluator.validate_predictions(item['predicoes'],r)
                if any(p['confianca']<.25 for p in item['predicoes']):raise ValueError('Cache compacto abaixo do limiar congelado.')
                predictions[model][cohort][r['id']]=item['predicoes']
    core.save_json(output/'fontes_autenticadas_19.json',{str(n):{'sha256':s['sha256'],'status':s['summary']['status']} for n,s in sources.items()})
    with gzip.open(output/'dados_validacao_reutilizados.json.gz','wt',encoding='utf-8') as f:json.dump(records,f,ensure_ascii=False)
    return {'sources':sources,'entries':all_entries,'references':references,'cohorts':cohorts,'predictions':predictions,'rule':rule}

def reproduce(bundle,policy,matcher,output):
    evaluations={m:{} for m in protocol.MODELS};checks=[]
    integer_fields=['tp_alerta','fn_alerta','fp_alerta','alta_ref','alta_sem_alerta','tp_detector','fn_detector',
                    'fp_detector','gt_avaliaveis','gt_ignorados','predicoes_ignoradas_conf']
    for model in protocol.MODELS:
        for cohort,records in bundle['cohorts'].items():
            result=evaluator.evaluate_cache(records,bundle['predictions'][model][cohort],.25,policy,matcher,Path(output)/'avaliacoes_reutilizadas'/model/cohort)
            original=bundle['references'][model][cohort];old_images={r['id']:r for r in original['images']}
            image_checks=0
            for row in result['images']:
                for k in integer_fields:
                    if row[k]!=int(old_images[row['id']][k]):raise ValueError('Contador por imagem divergente: '+model+'/'+cohort+'/'+row['id']+'/'+k)
                    image_checks+=1
            oldgt=data.gt_map(original['events']);newgt=data.gt_map(evaluator.parse_current_events(result['events']))
            if oldgt.keys()!=newgt.keys():raise ValueError('Referências GT divergentes.')
            for key,a in oldgt.items():
                b=newgt[key]
                if any(a[k]!=b[k] for k in ['evento','prioridade_ref','prioridade_pred','ha_alerta_ref','ha_alerta_pred']):
                    raise ValueError('Estado de objeto divergente na reprodução.')
            for k,v in original['summary']['alertas_por_objeto'].items():
                if result['summary']['alertas_por_objeto'][k]!=v:raise ValueError('Métrica de alerta divergente.')
            evaluations[model][cohort]=result
            checks.append({'modelo':model,'coorte':cohort,'checagens_imagem':image_checks,'objetos_GT':len(oldgt),'reproduziu':True})
            print('Previsões reaproveitadas e referência conferida:',model,cohort,flush=True)
    core.save_json(Path(output)/'reproducao_referencias_19.json',checks)
    return evaluations,checks
