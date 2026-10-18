"""Autenticação da auditoria, replay histórico e projeções locais por braço."""
from __future__ import annotations
import gzip,hashlib,json,shutil,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image
import yaml
import diag11_core as core
import diag11_data as d11
import control16_protocol as p16

QUARANTINE_IDS={
 'home:homeobjects_yolo/images/train/raw_train/living_room_1p (200).jpg',
 'home:homeobjects_yolo/images/train/raw_train/living_room_1p (203).jpg',
 'home:homeobjects_yolo/images/train/raw_train/living_room_590.jpg',
 'home:homeobjects_yolo/images/train/raw_train/living_room_718.jpg'}

def read_audit15(path,expected_sha):
    if core.sha256_file(path)!=expected_sha:raise ValueError('Pacote 15 diferente do analisado.')
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Pacote 15 com nomes repetidos.')
        for n in z.namelist():core.safe_member(n)
        if z.testzip():raise ValueError('CRC inválido no pacote 15.')
        summary=json.loads(z.read('resumo_auditoria_15.json'));cfg=json.loads(z.read('configuracao_auditoria_15.json'))
        records=json.loads(gzip.decompress(z.read('manifesto_imagens_auditadas.json.gz')))
    if not summary['concluido'] or len(records)!=6824 or p16.digest(records)!=cfg['dados_sha256']:
        raise ValueError('Auditoria 15 incompleta ou alterada.')
    if Counter(r['source'] for r in records)!={'home':1824,'openimages':5000} or any(r['role']!='train' for r in records):
        raise ValueError('Papel/origem inesperados no pacote 15.')
    by_id={r['id']:r for r in records}
    if len(by_id)!=6824 or not QUARANTINE_IDS<=by_id.keys():raise ValueError('Identidades da auditoria incompatíveis.')
    groups=defaultdict(list)
    for r in records:groups[r['sha256']].append(r)
    duplicates=[rs for rs in groups.values() if len(rs)>1]
    exact_ids={r['id'] for rs in duplicates for r in rs}
    if len(duplicates)!=2 or exact_ids!=QUARANTINE_IDS or any(len({r['label_sha256'] for r in rs})!=len(rs) for rs in duplicates):
        raise ValueError('Os pares exatos não correspondem à quarentena pré-definida.')
    return {'records':by_id,'summary':summary,'config':cfg,
            'quarantine':[{'id':r['id'],'image_sha256':r['sha256'],'label_sha256':r['label_sha256']}
                          for rs in duplicates for r in rs]}

def read_replay12(package_path,expected_sha,audit,expected_count=2000):
    if core.sha256_file(package_path)!=expected_sha:raise ValueError('Pacote 12 alterado.')
    with zipfile.ZipFile(package_path) as z:
        obj=json.loads(z.read('manifesto_replay.json'))
    rows=obj['imagens']
    if len(rows)!=expected_count or len({r['id'] for r in rows})!=expected_count:raise ValueError('Replay histórico incompleto.')
    for r in rows:
        a=audit['records']['oi:'+r['id']]
        if (r['sha256']!=a['sha256'] or r['label_sha256']!=a['label_sha256'] or r['known_mask']!=a['known_mask']
                or r['coverage_sha256']!=a['coverage_sha256'] or r['lote']!=a['lote']):
            raise ValueError('Replay 12 incompatível com a auditoria 15.')
    return rows

def extract_replay_exact(lots,rows,history_rows,local,fetch,cfg_sha,protected_ids,protected_hashes,near,audit):
    wanted={r['id']:r for r in rows};history={r['id']:r for r in history_rows};local=Path(local)
    records=[];seen=set();receipts=[]
    for item in lots:
        path=local/item['arquivo'];fetch(item,path)
        if core.sha256_file(path)!=item['zip_sha256'] or path.stat().st_size!=item['zip_bytes']:raise ValueError('Lote incompleto/alterado.')
        with zipfile.ZipFile(path) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Lote com nomes repetidos.')
            for n in z.namelist():core.safe_member(n)
            manifest=json.loads(z.read('manifesto_lote.json'))
            if manifest['config_sha256']!=cfg_sha or manifest['origem_split']!='train':raise ValueError('Lote fora do TRAIN autenticado.')
            for r in manifest['aceitas']:
                if r['id'] not in wanted:continue
                if r['id'] in seen:raise ValueError('ID duplicado no replay.')
                seen.add(r['id']);prior=wanted[r['id']];h=history[r['id']];a=audit['records']['oi:'+r['id']]
                if (h['split']!='train' or h['lote']!=item['arquivo'] or prior['lote']!=item['arquivo']
                    or h['imagem_sha256']!=r['sha256'] or int(h['known_mask'])!=r['known_mask']):raise ValueError('Replay fora da divisão histórica.')
                if r['id'] in protected_ids or r['sha256'] in protected_hashes:raise ValueError('Replay coincide com conjunto protegido.')
                mapping=[('imagem','sha256'),('label','label_sha256'),('cobertura','cobertura_sha256')]
                blobs={}
                for field,key in mapping:
                    raw=z.read(r[field]);sha=hashlib.sha256(raw).hexdigest()
                    expected=prior['coverage_sha256'] if key=='cobertura_sha256' else prior[key]
                    if sha!=r[key] or sha!=expected:raise ValueError('Bytes do replay divergentes.')
                    blobs[field]=raw
                cov=json.loads(blobs['cobertura'])
                positives=set(cov['classes_positivas_verificadas_exaustivas']);negatives=set(cov['classes_negativas_verificadas'])
                if (positives&negatives or sum(1<<c for c in positives|negatives)!=r['known_mask']
                    or cov['known_mask']!=r['known_mask']):raise ValueError('Cobertura parcial inválida.')
                image=local/'images/train'/('oi_'+r['id']+'.jpg');label=local/'labels/train'/('oi_'+r['id']+'.txt')
                for target,blob in [(image,blobs['imagem']),(label,blobs['label'])]:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
                with Image.open(image) as im:w,hgt=im.size;im.verify()
                if (w,hgt)!=(r['width'],r['height']):raise ValueError('Dimensões do replay alteradas.')
                if near.find(d11.phash(image)):raise ValueError('Replay histórico passou a coincidir perceptualmente com conjunto protegido; não haverá substituição silenciosa de IDs.')
                gt=core.parse_yolo(blobs['label'],w,hgt)
                if {g['classe_id'] for g in gt}&negatives or not positives<={g['classe_id'] for g in gt}:raise ValueError('Cobertura e caixas divergem.')
                if hashlib.sha256(blobs['label']).hexdigest()!=a['label_sha256']:raise ValueError('Label diverge da auditoria 15.')
                records.append({'id':'oi:'+r['id'],'oi_id':r['id'],'source':'openimages','role':'train',
                    'image':image,'label':label,'width':w,'height':hgt,'known_mask':r['known_mask'],
                    'sha256':r['sha256'],'label_sha256':r['label_sha256'],'coverage_sha256':r['cobertura_sha256'],
                    'lote':prior['lote'],'gt':gt,'name':image.name})
        receipts.append(item);path.unlink()
    if seen!=wanted.keys():raise ValueError('Não foram recuperadas todas as imagens do replay 12.')
    return records,receipts

def training_home(home,audit):
    selected=[];quarantined=[]
    for r in home:
        if r['role']!='home_train':continue
        a=audit['records'][r['id']]
        if r['sha256']!=a['sha256'] or r['label_sha256']!=a['label_sha256']:raise ValueError('Home diverge da auditoria 15.')
        if r['id'] in QUARANTINE_IDS:quarantined.append(r['id']);continue
        selected.append({**r,'role':'train','origin_role':'home_train','source':'home','known_mask':4095})
    if len(selected)!=1820 or set(quarantined)!=QUARANTINE_IDS:raise ValueError('Quarentena Home diferente da planejada.')
    return selected

def make_arms(records,home_val,destination,output):
    destination=Path(destination);output=Path(output);destination.mkdir(parents=True,exist_ok=True)
    if any(r['role']!='train' for r in records):raise ValueError('Somente TRAIN pode receber projeção de supervisão.')
    ids={r['id'] for r in records}
    if len(ids)!=len(records):raise ValueError('Treino com ID duplicado.')
    arms={};manifest=[];frozen=[]
    required=sum(Path(r['image']).stat().st_size for r in records)*2+256*1024**2
    if shutil.disk_usage(destination).free<required:raise RuntimeError('Espaço insuficiente para as cópias isoladas dos braços.')
    val_file=destination/'home_val.txt';val_file.write_text('\n'.join(str(Path(r['image']).resolve()) for r in home_val)+'\n')
    for arm in p16.ARMS:
        root=destination/arm;mapping={};rows=[];ignored=0
        for r in p16.logical_order(records):
            image=root/'images/train'/p16.filename(r);label=root/'labels/train'/Path(p16.filename(r)).with_suffix('.txt')
            image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(r['image'],image)
            original=Path(r['label']).read_bytes()
            if hashlib.sha256(original).hexdigest()!=r['label_sha256']:raise ValueError('Label original modificado antes da projeção.')
            projected,mask,removed=p16.project_label(original,r['source'],arm,r['known_mask']);ignored+=removed
            label.write_bytes(projected)
            if core.sha256_file(image)!=r['sha256']:raise ValueError('A cópia de imagem diverge do original.')
            if not p16.same_other_classes(original,projected):raise ValueError('Outras classes foram alteradas pela projeção.')
            gt=core.parse_yolo(projected,r['width'],r['height'])
            obj={**r,'image':image,'label':label,'gt':gt,'known_mask':mask,'original_known_mask':r['known_mask'],
                 'original_label_sha256':r['label_sha256'],'label_sha256':hashlib.sha256(projected).hexdigest(),
                 'ignored_home7_objects':removed,'arm':arm}
            if arm==p16.ARMS[1] and r['source']=='home' and (mask&(1<<7) or any(g['classe_id']==7 for g in gt)):
                raise ValueError('Wardrobe Home permaneceu supervisionada no braço experimental.')
            key=image.resolve().as_posix();mapping[key]=obj
            row={k:obj.get(k) for k in ['id','source','role','sha256','label_sha256','original_label_sha256',
                    'known_mask','original_known_mask','ignored_home7_objects']}
            row.update(arm=arm,relative_image=str(image.relative_to(destination)),relative_label=str(label.relative_to(destination)),
                       gt_per_class=dict(Counter(g['classe_id'] for g in gt)))
            rows.append(row);frozen.extend([(image,r['sha256']),(label,obj['label_sha256'])])
        train=root/'train.txt';train.write_text('\n'.join(sorted(mapping))+'\n')
        data_file=root/'data.yaml';data_file.write_text(yaml.safe_dump({'path':str(root.resolve()),'train':str(train.resolve()),
            'val':str(val_file.resolve()),'names':dict(enumerate(core.NAMES))},sort_keys=False,allow_unicode=True))
        arms[arm]={'records':mapping,'yaml':data_file,'manifest':rows,'ignored_home7_objects':ignored}
        manifest.extend(rows)
    a,b=[arms[k] for k in p16.ARMS]
    if sorted(r['id'] for r in a['records'].values())!=sorted(r['id'] for r in b['records'].values()):raise ValueError('Braços com imagens distintas.')
    core.save_json(output/'manifesto_supervisao_16.json',manifest)
    core.save_json(output/'quarentena_duplicacoes_exatas.json',sorted(QUARANTINE_IDS))
    return arms,frozen
