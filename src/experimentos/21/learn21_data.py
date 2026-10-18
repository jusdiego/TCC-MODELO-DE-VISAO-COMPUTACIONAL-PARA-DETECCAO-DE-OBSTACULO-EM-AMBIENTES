"""Preparação automática de TRAIN ampliado e das validações já expostas."""
from __future__ import annotations
from collections import Counter
import gzip,hashlib,json,math,shutil,urllib.request,zipfile
from pathlib import Path
from PIL import Image
import yaml
import diag11_core as core
import diag11_data as d11
from learn21_protocol import SPEC,digest,choose_candidates,balanced_pool

QUARANTINE={'home:homeobjects_yolo/images/train/raw_train/living_room_1p (200).jpg',
 'home:homeobjects_yolo/images/train/raw_train/living_room_1p (203).jpg',
 'home:homeobjects_yolo/images/train/raw_train/living_room_590.jpg',
 'home:homeobjects_yolo/images/train/raw_train/living_room_718.jpg'}

def preparation_sha(config):
    # Preserva a serialização canônica compacta da preparação histórica 10.
    return hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def safe_zip(path):
    z=zipfile.ZipFile(path)
    if len(z.namelist())!=len(set(z.namelist())):z.close();raise ValueError('ZIP com nomes repetidos.')
    for name in z.namelist():core.safe_member(name)
    return z

def checked_write(root,name,raw,sha):
    core.safe_member(name)
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Conteúdo alterado: '+name)
    p=Path(root)/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.is_file() or core.sha256_file(p)!=sha:p.write_bytes(raw)
    return p

def public_weight(target,expected):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file() and core.sha256_file(target)==expected['pretrained_sha256']:return target
    tmp=target.with_name(target.name+'.part')
    with urllib.request.urlopen(expected['pretrained_url'],timeout=60) as response,tmp.open('wb') as f:
        shutil.copyfileobj(response,f)
    if core.sha256_file(tmp)!=expected['pretrained_sha256']:
        tmp.unlink(missing_ok=True);raise ValueError('O peso público YOLOv8s difere do arquivo verificado.')
    tmp.replace(target);return target

def source19(package,expected,output):
    if core.sha256_file(package)!=expected['pacote_19']:raise ValueError('Pacote 19 diferente do analisado.')
    with safe_zip(package) as z:
        summary=json.loads(z.read('resumo_consolidacao_19.json'))
        if not summary['concluido']:raise ValueError('Fonte 19 incompleta.')
        records=json.loads(gzip.decompress(z.read('dados_validacao_reutilizados.json.gz')))
        if Counter(r['role'] for r in records)!={'home_val':456,'val_nova':1000}:raise ValueError('Papéis da validação alterados.')
        p=Path(output)/'referencia19';p.mkdir(parents=True,exist_ok=True)
        for name in ['comparacao_finais_e_referencias.csv','texto_base_capitulo4.md']:(p/name).write_bytes(z.read(name))
        for cohort in ['home_val_historica','openimages_val_nova']:
            (p/(cohort+'_candidato10.json')).write_bytes(z.read('avaliacoes_reutilizadas/candidato_10/'+cohort+'/metricas.json'))
    return records

def dataset_protection(archive,expected):
    if core.sha256_file(archive)!=expected['zip_novo']:raise ValueError('ZIP externo diferente do preservado.')
    with safe_zip(archive) as z:meta=json.loads(z.read('manifesto_validacao_nova.json'))
    claimed=meta['dataset_sha256']
    if claimed!=expected['dataset_novo'] or digest({k:v for k,v in meta.items() if k!='dataset_sha256'})!=claimed:
        raise ValueError('Manifesto externo alterado.')
    if Counter(r['role'] for r in meta['arquivos'])!={'val_nova':1000,'reserva_futura':300}:raise ValueError('Divisão externa inesperada.')
    # Da antiga reserva, somente identidade, bytes e pHash protegem a exclusão.
    protected=[{k:r[k] for k in ['id','sha256','phash']} for r in meta['arquivos']]
    return meta,protected

def materialise_validation(archive,meta,development,root):
    expected={r['id']:r for r in development if r['role']=='val_nova'};result=[]
    with safe_zip(archive) as z:
        for item in meta['arquivos']:
            if item['role']!='val_nova':continue
            if item['id'] not in expected:raise ValueError('ID externo fora da validação anterior.')
            r=expected[item['id']]
            for key in ['sha256','label_sha256','known_mask','width','height']:
                if r[key]!=item[key]:raise ValueError('Referência externa alterada.')
            for field,hfield in [('image','sha256'),('label','label_sha256'),('coverage','coverage_sha256')]:
                if not item[field].startswith({'image':'images','label':'labels','coverage':'coverage'}[field]+'/val_nova/'):
                    raise ValueError('Arquivo fora da validação de desenvolvimento.')
                p=checked_write(root,'external/'+item[field],z.read(item[field]),item[hfield])
                if field=='image':image=p
            result.append({**r,'image':image,'source':'validation','role':'val_nova'})
    if len(result)!=1000:raise ValueError('Validação externa incompleta.')
    return sorted(result,key=lambda r:r['id'])

def prepare_home(archive,expected,development,root,near,forbidden_sha):
    if core.sha256_file(archive)!=expected['home_zip']:raise ValueError('ZIP Home alterado.')
    previous={r['id']:r for r in development if r['role']=='home_val'}
    train=[];val=[];excluded=[];all_hashes=set()
    with safe_zip(archive) as z:
        manifest=json.loads(z.read('verificacao_preparacao.json'))
        if manifest['test']['sha256']!=expected['test_home'] or manifest['test']['imagens']!=403:raise ValueError('Home TEST diferente.')
        if {int(k):v for k,v in manifest['classes'].items()}!=dict(enumerate(core.NAMES)):raise ValueError('Classes Home diferentes.')
        entries={r['arquivo']:r for r in manifest['arquivos']}
        for name,item in sorted(entries.items()):
            if '/images/' not in name or Path(name).suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}:continue
            all_hashes.add(item['sha256'])
            role='train' if '/images/train/raw_train/' in name else 'home_val' if '/images/val/raw_train/' in name else None
            if role is None:continue
            iid='home:'+name
            if iid in QUARANTINE:excluded.append({'id':iid,'motivo':'quarentena histórica'});continue
            label_name=name.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
            token=hashlib.sha256(iid.encode()).hexdigest()[:24];folder='train' if role=='train' else 'val'
            image=checked_write(root,f'home/images/{folder}/{token}{Path(name).suffix}',z.read(name),item['sha256'])
            label=checked_write(root,f'home/labels/{folder}/{token}.txt',z.read(label_name),entries[label_name]['sha256'])
            with Image.open(image) as im:w,h=im.size;im.verify()
            ph=d11.phash(image)
            if role=='train' and (item['sha256'] in forbidden_sha or near.find(ph)):
                excluded.append({'id':iid,'motivo':'coincidência com validação ou conjunto protegido'});continue
            r={'id':iid,'image':image,'label':label,'name':image.name,'width':w,'height':h,'known_mask':4095,
                'sha256':item['sha256'],'label_sha256':entries[label_name]['sha256'],'phash':ph,
                'gt':core.parse_yolo(label.read_bytes(),w,h),'role':role,'source':'home','coorte':'home_val_historica' if role=='home_val' else 'train'}
            if role=='home_val':
                if iid not in previous or previous[iid]['sha256']!=r['sha256'] or previous[iid]['label_sha256']!=r['label_sha256']:raise ValueError('Home VAL alterada.')
                r['gt']=previous[iid]['gt'];val.append(r)
            else:train.append(r)
    if len(val)!=456 or len(train)<500:raise ValueError('Home insuficiente após a separação automática.')
    return train,sorted(val,key=lambda r:r['id']),excluded,all_hashes

def prepare_oi(packages,history,expected,root,drive,near,forbidden_ids,forbidden_sha,spec=SPEC,progress=None):
    candidates=choose_candidates(history,forbidden_ids,forbidden_sha,spec);wanted={r['id']:r for r in candidates}
    records=[];excluded=[];seen=set();receipts=[]
    lots=[item for item in packages['lotes'] if item['arquivo'] in {r['lote'] for r in candidates}]
    for li,item in enumerate(sorted(lots,key=lambda x:x['arquivo']),1):
        path=Path(root)/'downloads'/item['arquivo']
        print(f'Lote de preparação {li}/{len(lots)}: {item["arquivo"]}',flush=True)
        drive.fetch(expected['oi_root']+'/lotes/'+item['arquivo'],path,item['zip_sha256'])
        if path.stat().st_size!=item['zip_bytes']:raise ValueError('Tamanho do lote alterado.')
        with safe_zip(path) as z:
            meta=json.loads(z.read('manifesto_lote.json'))
            if meta['config_sha256']!=expected['preparacao'] or meta['origem_split']!='train':raise ValueError('Lote fora do TRAIN.')
            for item2 in meta['aceitas']:
                iid=item2['id']
                if iid not in wanted:continue
                prior=wanted[iid]
                if iid in seen or prior['split']!='train' or prior['lote']!=item['arquivo'] or prior['imagem_sha256']!=item2['sha256'] or int(prior['known_mask'])!=item2['known_mask']:
                    raise ValueError('Identidade ou divisão incompatível com a seleção.')
                seen.add(iid)
                if iid in forbidden_ids or item2['sha256'] in forbidden_sha:raise ValueError('TRAIN coincide com dado protegido.')
                raw={k:z.read(item2[k]) for k in ['imagem','label','cobertura']}
                for field,hkey in [('imagem','sha256'),('label','label_sha256'),('cobertura','cobertura_sha256')]:
                    if hashlib.sha256(raw[field]).hexdigest()!=item2[hkey]:raise ValueError('Arquivo do lote alterado.')
                cov=json.loads(raw['cobertura']);pos=set(cov['classes_positivas_verificadas_exaustivas']);neg=set(cov['classes_negativas_verificadas'])
                if pos&neg or sum(1<<c for c in pos|neg)!=item2['known_mask'] or cov['known_mask']!=item2['known_mask']:raise ValueError('Cobertura parcial inválida.')
                image=checked_write(root,'oi/images/train/oi_'+iid+'.jpg',raw['imagem'],item2['sha256'])
                label=checked_write(root,'oi/labels/train/oi_'+iid+'.txt',raw['label'],item2['label_sha256'])
                coverage=checked_write(root,'oi/coverage/train/oi_'+iid+'.json',raw['cobertura'],item2['cobertura_sha256'])
                with Image.open(image) as im:w,h=im.size;im.verify()
                if (w,h)!=(item2['width'],item2['height']):raise ValueError('Dimensões do treino alteradas.')
                ph=d11.phash(image)
                if near.find(ph):excluded.append({'id':iid,'motivo':'semelhança pHash com dados protegidos ou treino já aceito'});continue
                gt=core.parse_yolo(raw['label'],w,h)
                if {g['classe_id'] for g in gt}&neg or not pos<={g['classe_id'] for g in gt}:raise ValueError('Cobertura parcial e caixas divergem.')
                near.add(ph,iid)
                records.append({'id':'oi:'+iid,'oi_id':iid,'name':image.name,'role':'train','source':'openimages','coorte':'train',
                    'image':image,'label':label,'coverage':coverage,'width':w,'height':h,'known_mask':item2['known_mask'],
                    'sha256':item2['sha256'],'label_sha256':item2['label_sha256'],'coverage_sha256':item2['cobertura_sha256'],
                    'phash':ph,'gt':gt,'lote':item['arquivo']})
        receipts.append({'arquivo':item['arquivo'],'sha256':item['zip_sha256']});path.unlink()
        if progress:progress(li,len(lots))
    if seen!=set(wanted):raise ValueError('Algum ID planejado não foi localizado nos lotes autenticados.')
    return balanced_pool(records,spec),excluded,receipts

def input_files(records,cohorts,root):
    root=Path(root)
    train=root/'train.txt';val=root/'home_val.txt'
    train.write_text('\n'.join(str(Path(r['image']).resolve()) for r in records)+'\n')
    val.write_text('\n'.join(str(Path(r['image']).resolve()) for r in cohorts['home_val_historica'])+'\n')
    target=root/'data.yaml';target.write_text(yaml.safe_dump({'path':str(root.resolve()),'train':str(train.resolve()),'val':str(val.resolve()),'names':dict(enumerate(core.NAMES))},sort_keys=False))
    return target,{Path(r['image']).resolve().as_posix():r for r in records}

def freeze_base(records,cohorts,root,plan_key,report):
    root=Path(root)
    def serial(r):
        obj=core.json_clean(r)
        for key in ['image','label','coverage']:
            if key in r:obj[key]=Path(r[key]).relative_to(root).as_posix()
        return obj
    snapshot={'version':'21-base-v1','plan_key':plan_key,'train':[serial(r) for r in records],
        'cohorts':{c:[serial(r) for r in rows] for c,rows in cohorts.items()},'report':report}
    core.save_json(root/'base21_manifesto.json',snapshot)
    archive=root.parent/'base_treino_21.zip'
    with zipfile.ZipFile(archive.with_suffix('.tmp'),'w',zipfile.ZIP_STORED) as z:
        def store(path,name):
            # Datas dos arquivos não mudam a identidade da base numa nova sessão.
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
            info.external_attr=0o100644<<16
            with Path(path).open('rb') as source,z.open(info,'w',force_zip64=True) as target:shutil.copyfileobj(source,target,2**20)
        store(root/'base21_manifesto.json','base21_manifesto.json')
        paths=set()
        for r in records+[r for rows in cohorts.values() for r in rows]:
            paths.update(Path(r[k]) for k in ['image','label','coverage'] if k in r)
        for p in sorted(paths):store(p,p.relative_to(root).as_posix())
    archive.with_suffix('.tmp').replace(archive)
    return archive,snapshot

def restore_base(archive,root,plan_key):
    root=Path(root)
    with safe_zip(archive) as z:
        meta=json.loads(z.read('base21_manifesto.json'))
        if meta['plan_key']!=plan_key:raise ValueError('Base preparada pertence a outro plano.')
        rows=meta['train']+[r for group in meta['cohorts'].values() for r in group]
        for r in rows:
            if r['role'] not in {'train','home_val','val_nova'}:raise ValueError('Conjunto protegido presente na base preparada.')
            for field,hkey in [('image','sha256'),('label','label_sha256'),('coverage','coverage_sha256')]:
                if field in r:checked_write(root,r[field],z.read(r[field]),r[hkey])
    def hydrate(r):return {**r,**{k:root/r[k] for k in ['image','label','coverage'] if k in r}}
    train=[hydrate(r) for r in meta['train']];cohorts={c:[hydrate(r) for r in rows] for c,rows in meta['cohorts'].items()}
    return train,cohorts,meta['report']
