"""Fontes conhecidas, base cacheada e preparação repetível sem rótulos manuais."""
from pathlib import Path
from collections import Counter
import json,shutil
import diag11_core as core
import diag11_data as d11
import learn21_data as data
from learn21_protocol import SPEC,digest

def prepare(drive,expected,local,output,spec=SPEC):
    local=Path(local);output=Path(output);entries=local/'entradas';root=local/'base';root.mkdir(parents=True,exist_ok=True)
    plan={k:expected[k] for k in ['home_zip','test_home','preparacao','divisao_10','manifesto_pacotes','dataset_novo','zip_novo','pacote_19']}
    plan.update({k:spec[k] for k in ['oi_pool_max','oi_pool_min','oi_candidates','per_primary_class','seed']})
    plan['versao_preparacao']='21-v1';plan_key=digest(plan);remote='datasets/treino21_'+plan_key[:24]
    package=drive.fetch(expected['pasta_19']+'/pacote_consolidacao_19.zip',entries/'pacote19.zip',expected['pacote_19'])
    development=data.source19(package,expected,output)
    prepared_meta=drive.fetch(remote+'/base_pronta.json',entries/'base_pronta.json',optional=True)
    if prepared_meta:
        meta=json.loads(Path(prepared_meta).read_text())
        if meta['plan_key']!=plan_key:raise ValueError('Base remota pronta pertence a outro plano.')
        archive=drive.fetch(remote+'/base_treino_21.zip',local/'base_treino_21.zip',meta['zip_sha256'])
        records,cohorts,report=data.restore_base(archive,root,plan_key)
        core.save_json(output/'preparacao_dados_21.json',{**report,'base_reutilizada':True,'base_zip_sha256':meta['zip_sha256']})
        yaml,mapping=data.input_files(records,cohorts,root)
        return records,cohorts,yaml,meta['zip_sha256']
    prep=drive.fetch(expected['oi_root']+'/configuracao_preparacao.json',entries/'configuracao_preparacao.json')
    cfg=json.loads(Path(prep).read_text())
    if data.preparation_sha(cfg)!=expected['preparacao'] or cfg['nomes']!=core.NAMES:raise ValueError('Configuração Open Images alterada.')
    package_manifest=drive.fetch(expected['oi_root']+'/manifesto_pacotes.json',entries/'manifesto_pacotes.json',expected['manifesto_pacotes'])
    packages=json.loads(Path(package_manifest).read_text())
    if packages['config_sha256']!=expected['preparacao']:raise ValueError('Manifesto de lotes incompatível.')
    csv=drive.fetch(expected['pasta10']+'/divisao_openimages.csv',entries/'divisao.csv')
    js=drive.fetch(expected['pasta10']+'/divisao_openimages.json',entries/'divisao.json')
    history=d11.load_history(csv,js,expected['divisao_10'])
    archive_val=drive.fetch(expected['dataset_url'],entries/'validacao11.zip',expected['zip_novo'])
    metadata,protected=data.dataset_protection(archive_val,expected)
    near=d11.NearIndex()
    for r in protected:near.add(int(r['phash'],16),r['id'])
    for r in development:near.add(int(r['phash']),r['id'])
    protected_sha={r['sha256'] for r in development}|{r['sha256'] for r in protected}|{r['imagem_sha256'] for r in history['rows'] if r['split']=='test'}
    protected_ids={r['id'] for r in protected}|{r['id'] for r in development}
    external=data.materialise_validation(archive_val,metadata,development,root)
    home_zip=drive.fetch('datasets/homeobjects_sunrgbd_validado.zip',entries/'home.zip',expected['home_zip'])
    home,home_val,ex_home,home_hashes=data.prepare_home(home_zip,expected,development,root,near,protected_sha)
    for r in home:near.add(r['phash'],r['id'])
    oi,ex_oi,lots=data.prepare_oi(packages,history['rows'],expected,root,drive,near,protected_ids,protected_sha|home_hashes,spec)
    records=sorted(home+oi,key=lambda r:r['id']);cohorts={'home_val_historica':home_val,'openimages_val_nova':external}
    if {r['id'] for r in records}&protected_ids:raise ValueError('ID protegido entrou na preparação de treino.')
    counts=Counter(g['classe_id'] for r in records for g in r['gt'])
    absent=[core.NAMES[c] for c in range(12) if not counts[c]]
    report={'train_home':len(home),'train_openimages':len(oi),'train_total':len(records),'val_home':len(home_val),'val_external':len(external),
        'selecionado_do_train_80k':True,'historico_test20k_extraido':False,'home_test_extraido':False,'reserva20_extraida':False,
        'teste_final_novo_consultado':False,'base_reutilizada':False,'quarentena_home':sorted(data.QUARANTINE),
        'exclusoes_home':ex_home,'exclusoes_oi':ex_oi,'caixas_por_classe':{core.NAMES[c]:counts[c] for c in range(12)},
        'classes_sem_exemplo':absent,'lotes_lidos':lots,'plan_key':plan_key,
        'contexto':'Seleção por anotações humanas existentes, sem previsões ou relabeling; preserva cobertura parcial.'}
    core.save_json(output/'preparacao_dados_21.json',report)
    if absent:raise ValueError('Não há exemplos de treino para as classes: '+', '.join(absent))
    archive,_=data.freeze_base(records,cohorts,root,plan_key,report);base_sha=core.sha256_file(archive)
    ready=local/'base_pronta.json';core.save_json(ready,{'plan_key':plan_key,'zip_sha256':base_sha,'plan':plan})
    # Dois arquivos; o recibo só é publicado após a cópia completa do ZIP.
    print('Salvando a base preparada para retomadas...',flush=True)
    result=drive._call(['copyto',str(archive),drive.root+'/'+remote+'/base_treino_21.zip'])
    if not result.returncode:drive._call(['copyto',str(ready),drive.root+'/'+remote+'/base_pronta.json'])
    else:print('Base pronta localmente. A cópia de cache no Drive não foi concluída.',flush=True)
    core.save_json(output/'base_pronta_21.json',{'plan_key':plan_key,'zip_sha256':base_sha,'remote':remote})
    yaml,mapping=data.input_files(records,cohorts,root)
    return records,cohorts,yaml,base_sha
