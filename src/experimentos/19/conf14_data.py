"""Entradas autenticadas do diagnóstico 14; extrai somente Home VAL."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
from PIL import Image
import diag11_core as core

COHORTS = ('home_val_historica', 'openimages_val_nova')
BOOLS = {'True':True, 'False':False}

def rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))

def parse_events(raw):
    result = rows(raw)
    for r in result:
        r['classe_id'] = int(r['classe_id'])
        for field in ['bbox_ref','bbox_pred']:
            r[field] = json.loads(r[field]) if r[field] else None
        for field in ['ha_alerta_ref','ha_alerta_pred']:
            r[field] = BOOLS[r[field]]
        r['confianca'] = float(r['confianca']) if r['confianca'] else None
    return result

def gt_key(r):
    return (r['id'],int(r['classe_id']),tuple(round(float(v),6) for v in r['bbox_ref']))

def gt_map(events):
    result = {}
    for r in events:
        if r['bbox_ref'] is None: continue
        key = gt_key(r)
        if key in result: raise ValueError('Objeto GT repetido nos registros.')
        result[key] = r
    return result

def load_source12(path, expected, output=None):
    if core.sha256_file(path)!=expected['pacote_12']:
        raise RuntimeError('O pacote 12 mudou em relação ao recebido.')
    bundle = {'evaluations':{}}
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):
            raise RuntimeError('Pacote 12 com membros repetidos.')
        for name in z.namelist(): core.safe_member(name)
        if z.testzip() is not None: raise RuntimeError('Falha de CRC no pacote 12.')
        cfg = json.loads(z.read('configuracao_experimento_12.json'))
        summary = json.loads(z.read('resumo_experimento_12.json'))
        if not summary['concluido'] or summary['reserva_futura_inferida'] or summary['test_home_inferido'] or summary['test_oi20k_inferido']:
            raise RuntimeError('Execução 12 incompatível com o protocolo.')
        if cfg['expected']['candidato']!=expected['candidato'] or cfg['expected']['politica']!=expected['politica'] or summary['dataset_sha256']!=expected['dataset_novo']:
            raise RuntimeError('Referências ou política diferentes do experimento 12.')
        if cfg['codigo_embutido_sha256']!=expected['codigo_12']:
            raise RuntimeError('Código do experimento 12 diferente do recebido.')
        for name, digest in expected['codigo_12'].items():
            if hashlib.sha256(z.read('codigo/'+name)).hexdigest()!=digest:
                raise RuntimeError('Código alterado no pacote 12: '+name)
        rule_raw = z.read('regra_selecao_congelada.json')
        if hashlib.sha256(rule_raw).hexdigest()!=cfg['regra_sha256']:
            raise RuntimeError('A regra original do experimento 12 foi alterada.')
        selection = json.loads(z.read('selecao_checkpoints.json'))
        checkpoint = next(e for e in selection['assessed'] if e['id']=='B_home_replay_e03')
        if checkpoint['weight_sha256']!=expected['B_e03']:
            raise RuntimeError('Checkpoint B_e03 diferente do esperado.')
        for model,prefix in [('baseline','referencias/baseline'),('candidato_10','referencias/candidato_10'),('B_e03','avaliacoes/B_home_replay_e03')]:
            bundle['evaluations'][model] = {}
            for cohort in COHORTS:
                base=prefix+'/'+cohort+'/'
                members={name:z.read(base+name) for name in ['metricas.json','alertas_por_imagem.csv','avaliacao_objetos.csv']}
                metrics=json.loads(members['metricas.json'])
                images=rows(members['alertas_por_imagem.csv'])
                for r in images:
                    for key in ['tp_alerta','fn_alerta','fp_alerta','alta_ref','alta_sem_alerta','tp_detector','fn_detector','fp_detector','gt_avaliaveis','gt_ignorados','predicoes_ignoradas_conf_025']:
                        r[key]=int(r[key])
                if len({r['id'] for r in images})!=len(images) or len(images)!=metrics['imagens']:
                    raise RuntimeError('IDs ou quantidades inconsistentes nos registros 12.')
                for field, n in [('tp_alerta',metrics['alertas_por_objeto']['tp']),('fn_alerta',metrics['alertas_por_objeto']['fn']),('fp_alerta',metrics['alertas_por_objeto']['fp']),('alta_ref',metrics['alta_ref']),('alta_sem_alerta',metrics['alta_sem_alerta'])]:
                    if sum(r[field] for r in images)!=n: raise RuntimeError('Contagens inconsistentes no pacote 12.')
                events=parse_events(members['avaliacao_objetos.csv'])
                gt_map(events)
                bundle['evaluations'][model][cohort]={'summary':metrics,'images':images,'events':events}
                if output is not None:
                    dest=Path(output)/'referencias_12'/model/cohort
                    dest.mkdir(parents=True,exist_ok=True)
                    for name, raw in members.items(): (dest/name).write_bytes(raw)
        for cohort in COHORTS:
            target=gt_map(bundle['evaluations']['B_e03'][cohort]['events'])
            for model in ['baseline','candidato_10']:
                source=gt_map(bundle['evaluations'][model][cohort]['events'])
                if source.keys()!=target.keys(): raise RuntimeError('Modelos do 12 usam referências GT diferentes.')
                if any(source[k]['ha_alerta_ref']!=target[k]['ha_alerta_ref'] or source[k]['prioridade_ref']!=target[k]['prioridade_ref'] for k in target):
                    raise RuntimeError('Referências de prioridade diferentes entre modelos.')
        bundle.update(config=cfg,summary=summary,rule=json.loads(rule_raw),
                      audit=json.loads(z.read('auditoria_dados.json')))
    return bundle

def extract_home_val(path, destination, expected_zip, expected_test, expected_policy):
    if core.sha256_file(path)!=expected_zip: raise RuntimeError('ZIP Home alterado.')
    destination=Path(destination)
    records=[];frozen=[]
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())): raise RuntimeError('Membros repetidos no Home.')
        for name in z.namelist():core.safe_member(name)
        manifest=json.loads(z.read('verificacao_preparacao.json'))
        if (manifest['test']['sha256']!=expected_test or manifest['test']['imagens']!=403 or
            {int(k):v for k,v in manifest['classes'].items()}!=dict(enumerate(core.NAMES))):
            raise RuntimeError('Manifesto Home incompatível.')
        root=manifest['dataset_dir']
        items={r['arquivo']:r for r in manifest['arquivos']}
        if len(items)!=len(manifest['arquivos']): raise RuntimeError('Manifesto Home repetido.')
        train_hash=set();val_hash=set();all_hash=set();counts=Counter()
        for name,item in items.items():
            is_image=name.startswith(root+'/images/') and Path(name).suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}
            if not is_image:continue
            all_hash.add(item['sha256'])
            if name.startswith(root+'/images/train/raw_train/'):
                counts['train']+=1;train_hash.add(item['sha256'])
            elif name.startswith(root+'/images/val/raw_train/'):
                counts['val']+=1;val_hash.add(item['sha256'])
        if counts!={'train':1824,'val':456} or train_hash&val_hash:
            raise RuntimeError('A divisão atual de Home mudou ou tem bytes repetidos entre TRAIN e VAL.')
        for name in z.namelist():
            if not(name.startswith('codigo/') and name.endswith('.py')):continue
            raw=z.read(name)
            if name in items and hashlib.sha256(raw).hexdigest()!=items[name]['sha256']:
                raise RuntimeError('Código Home alterado.')
            out=destination/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
            frozen.append((out,hashlib.sha256(raw).hexdigest()))
        for name,item in sorted(items.items()):
            if not(name.startswith(root+'/images/val/raw_train/') and Path(name).suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}):continue
            label=name.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
            if label not in items:raise RuntimeError('Label Home ausente.')
            paths={}
            for field,member in [('image',name),('label',label)]:
                raw=z.read(member)
                if hashlib.sha256(raw).hexdigest()!=items[member]['sha256']:raise RuntimeError('Home alterado: '+member)
                out=destination/member;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
                paths[field]=out;frozen.append((out,items[member]['sha256']))
            with Image.open(paths['image']) as im:
                width,height=im.size;im.verify()
            records.append({'id':'home:'+name,'name':name.split('/raw_train/')[-1],**paths,
                'role':'home_val','coorte':'home_val_historica','width':width,'height':height,
                'known_mask':core.ALL_KNOWN,'sha256':item['sha256'],'label_sha256':items[label]['sha256'],
                'gt':core.parse_yolo(paths['label'].read_bytes(),width,height,source=label)})
    if len(records)!=456 or core.sha256_file(destination/'codigo/src/politica_alertas.py')!=expected_policy:
        raise RuntimeError('Quantidade Home VAL ou política V2 incompatível.')
    audit={'home_train_manifesto':1824,'home_val_extraida':456,'home_train_extraida':False,
           'home_test_extraido':False,'test_home_manifesto':403,'intersecao_bytes_train_val':0,
           'historico_original_baseline_nao_comprovado_pela_separacao_atual':True}
    return records,all_hash,frozen,audit

def validate_cohorts(cohorts,bundle,reserved_ids):
    if set(cohorts)!=set(COHORTS) or {k:len(v) for k,v in cohorts.items()}!={'home_val_historica':456,'openimages_val_nova':1000}:
        raise ValueError('Coortes diferentes das duas validações congeladas.')
    for cohort,records in cohorts.items():
        ids={r['id'] for r in records}
        if len(ids)!=len(records) or ids&set(reserved_ids):raise ValueError('ID repetido ou reservado na avaliação.')
        expected_role='home_val' if cohort=='home_val_historica' else 'val_nova'
        if any(r['role']!=expected_role for r in records):raise ValueError('Papel incompatível com a coorte de validação.')
        old=bundle['evaluations']['B_e03'][cohort]
        if ids!={r['id'] for r in old['images']}:raise ValueError('IDs diferentes dos avaliados no 12.')
        actual={}
        for r in records:
            for g in r['gt']:
                if r['known_mask']&(1<<g['classe_id']):
                    k=(r['id'],g['classe_id'],tuple(round(float(x),6) for x in g['bbox']))
                    if k in actual:raise ValueError('GT repetido na entrada atual.')
                    actual[k]=True
        if actual.keys()!=gt_map(old['events']).keys():raise ValueError('Caixas avaliáveis diferentes das referências originais.')
    return True
