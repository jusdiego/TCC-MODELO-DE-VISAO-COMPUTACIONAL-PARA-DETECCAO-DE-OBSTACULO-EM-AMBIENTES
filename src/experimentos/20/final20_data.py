"""Reserva final: somente os 300 IDs e as anotações do manifesto autenticado."""
from __future__ import annotations
from collections import Counter
import gzip,hashlib,json,math,zipfile
from pathlib import Path
from PIL import Image
import diag11_core as core

MODELS=('baseline','candidato_10')
ROLE='reserva_futura'
COHORT='openimages_reserva_final'

def digest(value):
    return hashlib.sha256(core.canonical(value).encode()).hexdigest()

def save_gz(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=4) as f:
        json.dump(core.json_clean(value),f,ensure_ascii=False,allow_nan=False)
    tmp.replace(path)

def source19(path,expected,frozen,output):
    if core.sha256_file(path)!=expected['pacote_19']:raise ValueError('O pacote 19 difere do resultado analisado.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        if len(set(z.namelist()))!=len(z.namelist()) or z.testzip():raise ValueError('Pacote 19 inválido.')
        for name in z.namelist():core.safe_member(name)
        summary=json.loads(z.read('resumo_consolidacao_19.json'))
        plan=json.loads(z.read('protocolo_avaliacao_final.json'))
        reg=json.loads(z.read('registro_protocolo_final.json'))
        if (not summary['concluido'] or summary['context_key']!=expected['contexto_19']
            or plan!=frozen or digest(plan)!=expected['protocolo_sha256']
            or reg['sha256_canonico']!=digest(plan) or reg['reserva_consultada']
            or summary['reserva_futura_extraida'] or summary['reserva_futura_inferida']):
            raise ValueError('O protocolo congelado no 19 não corresponde à avaliação planejada.')
        development=json.loads(gzip.decompress(z.read('dados_validacao_reutilizados.json.gz')))
        if Counter(r['role'] for r in development)!={'home_val':456,'val_nova':1000}:
            raise ValueError('Referências de desenvolvimento incompatíveis.')
        for name in ['protocolo_avaliacao_final.json','registro_protocolo_final.json',
                     'comparacao_finais_e_referencias.csv','resumo_hipoteses_16_18.csv']:
            target=output/'fontes19'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    return plan,development

def extract_reserve(archive,plan,development,destination,output):
    if core.sha256_file(archive)!=plan['archive_sha256']:raise ValueError('ZIP da reserva diferente do preservado.')
    destination=Path(destination);output=Path(output)
    records=[];receipts=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('ZIP com nomes repetidos.')
        for name in names:core.safe_member(name)
        meta=json.loads(z.read('manifesto_validacao_nova.json'))
        claimed=meta['dataset_sha256']
        if claimed!=plan['dataset_sha256'] or digest({k:v for k,v in meta.items() if k!='dataset_sha256'})!=claimed:
            raise ValueError('Manifesto da reserva foi alterado.')
        items=meta['arquivos'];counts=Counter(r['role'] for r in items)
        if (meta['nomes']!=core.NAMES or counts!={'val_nova':1000,ROLE:plan['images']}
            or meta['contagens']!=dict(counts) or len({r['id'] for r in items})!=len(items)):
            raise ValueError('Mapeamento, divisão ou IDs incompatíveis com o plano congelado.')
        dev={r['id']:r for r in development if r['role']=='val_nova'}
        frozen_val={r['id']:r for r in items if r['role']=='val_nova'}
        if set(dev)!=set(frozen_val):raise ValueError('A validação anterior não pertence a esta divisão.')
        for iid,r in frozen_val.items():
            for field in ['sha256','label_sha256','known_mask','width','height']:
                if r[field]!=dev[iid][field]:raise ValueError('A divisão anterior mudou: '+iid+'/'+field)
        selected=sorted((r for r in items if r['role']==ROLE),key=lambda r:r['id'])
        if len(selected)!=plan['images']:raise ValueError('A reserva não tem todos os 300 IDs previstos.')
        if {r['id'] for r in selected}&set(dev):raise ValueError('IDs da reserva apareceram na validação.')
        dev_sha={r['sha256'] for r in development}
        if any(r['sha256'] in dev_sha for r in selected):raise ValueError('Bytes da reserva já aparecem no desenvolvimento.')
        for index,item in enumerate(selected,1):
            iid=item['id']
            if (not isinstance(iid,str) or len(iid)!=16 or any(c not in '0123456789abcdef' for c in iid)):
                raise ValueError('ID externo inválido.')
            expected_members={'image':f'images/{ROLE}/oi_val_{iid}.jpg',
                              'label':f'labels/{ROLE}/oi_val_{iid}.txt',
                              'coverage':f'coverage/{ROLE}/oi_val_{iid}.json'}
            extracted={}
            for field,member in expected_members.items():
                if item[field]!=member:raise ValueError('Arquivo fora da reserva: '+item[field])
                raw=z.read(member);key='sha256' if field=='image' else field+'_sha256'
                if hashlib.sha256(raw).hexdigest()!=item[key]:raise ValueError('Bytes alterados: '+member)
                target=destination/member;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.is_file() or core.sha256_file(target)!=item[key]:target.write_bytes(raw)
                extracted[field]=target
                receipts.append({'id':iid,'papel':ROLE,'tipo':field,'member':member,'sha256':item[key]})
            with Image.open(extracted['image']) as im:
                if im.size!=(item['width'],item['height']):raise ValueError('Dimensões diferentes do manifesto.')
                im.verify()
            coverage=json.loads(extracted['coverage'].read_text())
            for key in ['known_mask','positive_mask','negative_mask']:
                if (type(item[key])!=int or not 0<=item[key]<=core.ALL_KNOWN or coverage[key]!=item[key]):
                    raise ValueError('Cobertura de referência divergente: '+iid)
            if item['positive_mask']&item['negative_mask'] or (item['positive_mask']|item['negative_mask'])!=item['known_mask']:
                raise ValueError('Máscaras de cobertura contraditórias.')
            ph=item['phash']
            if not isinstance(ph,str) or len(ph)!=16 or any(c not in '0123456789abcdef' for c in ph):
                raise ValueError('pHash64 congelado inválido.')
            gt=[]
            for g in item['gt_normalizado']:
                cid=g['classe_id'];box=g['bbox_norm']
                if (type(cid)!=int or not 0<=cid<len(core.NAMES) or g['classe']!=core.NAMES[cid]
                    or len(box)!=4 or not all(math.isfinite(v) for v in box)
                    or min(box)<0 or max(box)>1 or box[2]<=box[0] or box[3]<=box[1]):
                    raise ValueError('Caixa de referência inválida: '+iid)
                gt.append({**g,'bbox':[box[0]*item['width'],box[1]*item['height'],box[2]*item['width'],box[3]*item['height']]})
            parsed=core.parse_yolo(extracted['label'].read_bytes(),item['width'],item['height'])
            if len(parsed)!=len(gt) or any(a['classe_id']!=b['classe_id'] or max(abs(x-y) for x,y in zip(a['bbox'],b['bbox']))>1e-5 for a,b in zip(parsed,gt)):
                raise ValueError('Anotações YOLO divergem das caixas congeladas: '+iid)
            records.append({'id':iid,'name':extracted['image'].name,'image':extracted['image'],
                'label':extracted['label'],'coverage':extracted['coverage'],'role':ROLE,'coorte':COHORT,
                **{k:item[k] for k in ['width','height','known_mask','positive_mask','negative_mask','sha256','label_sha256','coverage_sha256']},
                'phash':int(ph,16),'phash_origem':'manifesto autenticado do notebook 11','gt':gt,'creditos':item.get('creditos',{})})
            if index%100==0:print(f'Reserva conferida: {index}/{len(selected)}',flush=True)
    near=[]
    for r in records:
        found=next((d for d in development if (r['phash']^int(d['phash'])).bit_count()<=4),None)
        if found:near.append({'id_reserva':r['id'],'id_desenvolvimento':found['id'],'distancia':(r['phash']^int(found['phash'])).bit_count()})
    audit={'imagens':len(records),'dataset_sha256':plan['dataset_sha256'],'archive_sha256':plan['archive_sha256'],
        'ids_substituidos':0,'imagens_excluidas':0,'intersecao_ids_val_anterior':0,'intersecao_bytes_desenvolvimento':0,
        'pares_semelhanca_phash_com_desenvolvimento':near,'semelhanca_phash_nao_prova_duplicacao':True,
        'train_extraido':False,'home_test_extraido':False,'oi_test20k_extraido':False,
        'origem_treino_original_baseline':'nao_determinado','recibos':receipts}
    core.save_json(output/'extracao_reserva_20.json',audit)
    save_gz(output/'registros_reserva_20.json.gz',[{k:v for k,v in r.items() if k not in {'image','label','coverage'}} for r in records])
    return records,audit
