"""Corrige a referência de contagem após o loader; preserva os treinos do 18."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib,inspect,json
from pathlib import Path
import numpy as np
import yaml
import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
import ultralytics.data.augment as augment_source
import diag11_core as core
import exp12_data as frozen_data
from exp12_training import isolated_evaluation
import mix18_protocol as protocol

VERSION='18-auditoria-pos-loader-1.0.1'

def counts(values):
    return dict(Counter(str(int(v)) for v in values))

def raw_rows(record):
    if core.sha256_file(record['label'])!=record['label_sha256']:
        raise ValueError('Rótulo original alterado: '+record['id'])
    rows=np.array([list(map(float,l.split())) for l in Path(record['label']).read_text(encoding='utf-8-sig').splitlines() if l.strip()],dtype=np.float32).reshape(-1,5)
    return np.unique(rows,axis=0)

def assert_pipeline(dataset):
    # Albumentations fotométricas não afetam a contagem; uma transformação espacial
    # externa impediria usar uma referência determinística por imagem.
    seen=set()
    def visit(transform):
        if id(transform) in seen:return
        seen.add(id(transform))
        if type(transform).__name__=='Albumentations' and getattr(transform,'contains_spatial',False):
            raise ValueError('Albumentations espacial inesperada na referência de alvos.')
        for child in getattr(transform,'transforms',[]) or []:visit(child)
        pre=getattr(transform,'pre_transform',None)
        if pre is not None:visit(pre)
    visit(dataset.transforms)

def loader_reference(records,data,arguments,output):
    """Repassa as imagens pelo mesmo dataset de TRAIN, sem modelo ou otimizador."""
    output=Path(output)
    if ultralytics.__version__!='8.4.150':raise RuntimeError('A correção requer ultralytics==8.4.150, como o treino.')
    if (arguments.get('rect') or arguments.get('workers')!=0 or arguments.get('cache')
        or any(float(arguments.get(k,0))!=0 for k in protocol.AUGMENTATION_ZERO)):
        raise ValueError('Parâmetros incompatíveis com a referência determinística do 18.')
    wanted={Path(r['image']).resolve().as_posix():r for r in records}
    if len(wanted)!=len(records) or len({r['id'] for r in records})!=len(records):raise ValueError('Imagens/IDs repetidos na referência.')
    if set(wanted)!=set(data['records']):raise ValueError('Dataset comum diferente dos registros autenticados.')
    cfg=get_cfg(overrides=deepcopy(arguments))
    with Path(data['yaml']).open() as f:dataset_data=yaml.safe_load(f)
    dataset_data['nc']=len(core.NAMES)
    names=dataset_data['names']
    names=dict(enumerate(names)) if isinstance(names,list) else {int(k):v for k,v in names.items()}
    if names!=dict(enumerate(core.NAMES)):raise ValueError('Mapeamento de classes mudou no dataset comum.')
    rows=[];differences=[]
    with isolated_evaluation():
        dataset=build_yolo_dataset(cfg,dataset_data['train'],cfg.batch,dataset_data,mode='train',rect=False,stride=32)
        assert_pipeline(dataset)
        paths=[Path(f).resolve().as_posix() for f in dataset.im_files]
        if len(paths)!=len(wanted) or set(paths)!=set(wanted):raise ValueError('O loader ignorou/adicionou uma imagem.')
        for index,key in enumerate(paths):
            record=wanted[key];original=raw_rows(record);label=dataset.labels[index]
            stored=np.concatenate([label['cls'],label['bboxes']],axis=1).astype(np.float32)
            canon=lambda a:sorted(tuple(float(x) for x in r) for r in a)
            if canon(original)!=canon(stored):raise ValueError('O cache bruto do loader diverge do rótulo autenticado: '+record['id'])
            sample=dataset[index]
            if Path(sample['im_file']).resolve().as_posix()!=key:raise ValueError('O loader trocou a identidade da imagem.')
            raw=counts(original[:,0]);processed=counts(sample['cls'].flatten().tolist())
            if any(v>raw.get(k,0) for k,v in processed.items()):raise ValueError('O processamento adicionou alvos inesperados: '+record['id'])
            row={'id':record['id'],'image_sha256':record['sha256'],'label_sha256':record['label_sha256'],
                 'known_mask':record['known_mask'],'raw_unique_counts':raw,'processed_counts':processed}
            rows.append(row)
            if raw!=processed:
                # O caso que disparou a auditoria deve ser reproduzível antes de aceitar.
                for _ in range(2):
                    again=dataset[index]
                    if counts(again['cls'].flatten().tolist())!=processed:
                        raise ValueError('Contagem de alvos não determinística: '+record['id'])
                differences.append({**row,'discarded_by_class':{k:v-processed.get(k,0) for k,v in raw.items() if v!=processed.get(k,0)}})
            if (index+1)%250==0 or index+1==len(paths):
                print(f'Referência após loader: {index+1}/{len(paths)} imagens',flush=True)
        del dataset
    summary={'versao':VERSION,'imagens':len(rows),'imagens_com_alvos_filtrados':len(differences),
        'alvos_filtrados_unicos':sum(sum(r['discarded_by_class'].values()) for r in differences),
        'ultralytics':ultralytics.__version__,'augment_source_sha256':core.sha256_file(inspect.getfile(augment_source)),
        'reference_sha256':protocol.digest(rows),'argumentos':arguments,
        'treino_alterado':False,'rotulos_alterados':False,'cobertura_alterada':False,
        'explicacao':'A contagem é comparada após o mesmo pipeline TRAIN. O filtro padrão de caixas pode remover alvos pequenos/degenerados; a cobertura e os bytes originais continuam obrigatórios.',
        'diferencas':differences}
    protocol.save_gz(output/'referencia_alvos_pos_loader.json.gz',rows)
    core.save_json(output/'diagnostico_alvos_pos_loader.json',summary)
    return {r['id']:r for r in rows},summary

def audit_with_loader(results,plans,records,specs,data,arguments,output,frozen=None):
    if frozen is not None:frozen_data.verify_frozen(frozen)
    # Os registros do treino comum preservam os IDs, mas apontam para as cópias
    # usadas de fato pelo loader. A auditoria usa essas cópias e confere os hashes.
    common=list(data['records'].values())
    original={r['id']:r for r in records}
    if len(original)!=len(records) or len(common)!=len(records):raise ValueError('Manifesto de treino incompleto.')
    for r in common:
        prior=original.get(r['id'])
        if prior is None or any(r[k]!=prior[k] for k in ['sha256','label_sha256','known_mask','source']):
            raise ValueError('Supervisão/cobertura comum alterada: '+r['id'])
    reference,summary=loader_reference(common,data,arguments,output)
    prior_counts=protocol.target_counts
    try:
        # Só muda a referência da auditoria. Nenhum loader de treino é executado
        # pelo modelo e nenhuma checagem de máscara, gradiente ou orçamento é removida.
        protocol.target_counts=lambda r:reference[r['id']]['processed_counts']
        result=protocol.audit_runs(results,plans,records,specs)
    except Exception as exc:
        divergences=[]
        for arm,run in results.items():
            for row in run['logs']:
                for j,iid in enumerate(row['ids']):
                    observed={str(k):v for k,v in row['gt_per_image_by_class'][j].items()}
                    expected=reference[iid]
                    if observed!=expected['processed_counts'] or row['known_masks'][j]!=expected['known_mask']:
                        divergences.append({'arm':arm,'step':row['step'],'id':iid,'observed_counts':observed,
                            'expected_counts':expected['processed_counts'],'observed_mask':row['known_masks'][j],
                            'expected_mask':expected['known_mask']})
        core.save_json(Path(output)/'divergencias_nao_explicadas.json',{'erro':str(exc),'quantidade':len(divergences),'divergencias':divergences})
        raise ValueError('A divergência não foi explicada pelo processamento padrão; checagens preservadas. O diagnóstico foi salvo no pacote.') from exc
    finally:protocol.target_counts=prior_counts
    result.update({'auditoria_versao':VERSION,'alvos_comparados_pos_loader':True,
        'treino_alterado_para_corrigir_auditoria':False,'imagens_com_alvos_filtrados':summary['imagens_com_alvos_filtrados'],
        'alvos_filtrados_unicos':summary['alvos_filtrados_unicos'],'referencia_loader_sha256':summary['reference_sha256']})
    core.save_json(Path(output)/'auditoria_fontes.json',result)
    core.save_json(Path(output)/'correcao_auditoria18.json',{'versao':VERSION,'concluida':True,
        'checagens_mascara_gradiente_orcamento_preservadas':True,'treinamento_modificado':False,
        'referencia':'referencia_alvos_pos_loader.json.gz','diagnostico':'diagnostico_alvos_pos_loader.json'})
    print('Auditoria corrigida e conferida. Imagens com filtro padrão de alvos:',summary['imagens_com_alvos_filtrados'],flush=True)
    return result

def recover(namespace,final_cell):
    required=['PASTA_RESULTADOS','PASTA_CODIGO','BRACOS','CRONOGRAMAS','REGISTROS_TREINO','SPECS',
        'ARGUMENTOS_TREINO','CONGELADOS','CONTEXT_KEY','INITIAL_STATE','estado18','registrar_falha']
    missing=[k for k in required if k not in namespace]
    if missing:raise RuntimeError('Execute esta célula no notebook 18 original, na mesma sessão que terminou a etapa 7. Variáveis ausentes: '+', '.join(missing))
    output=Path(namespace['PASTA_RESULTADOS'])
    try:
        results={}
        for arm in protocol.ARMS:
            result=namespace['estado18'].load_completed(output,arm,namespace['CONTEXT_KEY'],namespace['INITIAL_STATE'],
                namespace['SPECS'][arm],namespace['CRONOGRAMAS'][arm])
            if result is None:raise RuntimeError('Não foi possível autenticar o braço completo '+arm+'. A recuperação foi interrompida sem iniciar outro treino.')
            results[arm]=result
            print('Braço completo autenticado, sem repetir treino:',arm,flush=True)
        audit=audit_with_loader(results,namespace['CRONOGRAMAS'],namespace['REGISTROS_TREINO'],namespace['SPECS'],
            namespace['BRACOS'][protocol.ARMS[0]],namespace['ARGUMENTOS_TREINO'],output,namespace['CONGELADOS'])
        namespace['RESULTADOS_BRACOS']=results;namespace['PAREAMENTO']=audit
    except Exception as exc:
        namespace['registrar_falha'](exc,'correção da auditoria após loader');raise
    print('Gerando o relatório e o pacote final automaticamente...',flush=True)
    exec(compile(final_cell,'etapa_8_original_preservada','exec'),namespace)
