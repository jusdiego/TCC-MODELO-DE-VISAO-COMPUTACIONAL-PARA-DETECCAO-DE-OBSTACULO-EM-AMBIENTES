"""Identidades, protocolo e persistência do experimento 24."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import csv, gzip, hashlib, importlib.metadata, json, math, os, shutil, stat, time, urllib.request, zipfile
import numpy as np

VERSION='24-rfdetr-small-home-v1'
NAMES=['bed','sofa','chair','table','lamp','tv','laptop','wardrobe','window','door','potted plant','photo frame']
PT=['Cama','Sofá','Cadeira','Mesa','Luminária','TV','Laptop','Guarda-roupa','Janela','Porta','Planta em vaso','Quadro']
ALERT_IDS=[0,1,2,3,7,10]
SPEC={'version':VERSION,'model':'RFDETRSmall','resolution':512,'epochs':80,'batch_size':4,
 'effective_batch':16,'lr':1e-4,'lr_encoder':1.5e-4,'weight_decay':1e-4,'seed':24002026,
 'amp_dtype':'fp16','use_ema':True,'early_stopping_patience':12,'early_stopping_min_delta':.001,
 'training_minutes':140,'session_minutes':180,'evaluation_reserve_minutes':20,'pilot_epochs':3,
 'num_workers':2,'thresholds':[.05,.10,.15,.20,.25,.30,.40,.50,.60,.70,.80,.90],
 'capture_conf':.01,'match_iou':.5,'max_det':300,'target_precision':.8,'target_recall':.7,
 'selection_checkpoint':'mAP50-95 de VAL, EMA; empate preserva época anterior',
 'selection_threshold':'max(min(P/0.8,R/0.7)+1e-6*F1), empate menor confiança; somente VAL',
 'secondary_ids':ALERT_IDS,'secondary_threshold':'mesmo limiar do desfecho principal',
 'bootstrap_replicates':2000,'bootstrap_unit':'imagem; não há identificação validada de cenas Home',
 'test_status':'Home TEST histórico já consultado; avaliação desta rodada após congelamento',
 'external_policy':'NYU TEST opcional, sem ajuste, com exclusão de semelhança Home; não é benchmark independente garantido'}
EXPECTED={'home_sha':'ada7fbee37b83f673b1b76bc57643d1fe4648941e0e8a7b895282216aed47676',
 'home_test_sha':'f48be0a2b3a93e59259b1ae0d3655028951bf270ef7f4e7916a9424530951a89',
 'policy_sha':'1aa6b13cb68d8f30a2f388c3577a61a12dcac4c604581877091d3fbdc6042845',
 'matcher_sha':'1da5dff21a6c6f6ec53d041247ce18d012008c4b26023b52c3fd2745f4c4105d',
 'c10_sha':'97040f91876d8835a67a7db287d3e1c5f129a4068e36dd4483d947f62b260c5f',
 'c10_path':'resultados/yolov8n_home_openimages_80_20_1ep_kaggle_20261015_030228_068010/best.pt',
 '21_context':'5b35ca0c28f97027e7d1f324fb2ef384d1b723be04c6176f796270f92af1d6b4',
 '21_sha':'5d084e766be8c7a98ee93365e2d14e246dfe7d18cdcabd87df566ff39acc2ee3',
 '21_path':'resultados/treino21_5b35ca0c28f97027e7d1f324/pesos/epoca_021_inferencia.pt',
 'rfdetr_url':'https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth',
 'rfdetr_md5':'fb37061c1af7bace359c91b723a8d5c1',
 'rfdetr_sha256':'d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33',
 'nyu_id':'b174f1c72bb9403a5cc1498ffa7b5c38cfa1a5cdfc3c54a79a9b679724521fd1',
 'nyu_reserve_sha256':'64a25836d6665e9c80141ac9ae74c2c2b0fa455eb2289fe3a5fdf194ce317074',
 'nyu23_context':'e07346585506a2db3229645f72d287e7fa9680aa2ffed7b4ddc1514a8090d1b9'}

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('Valor não finito no resultado.')
    return x
def canonical(x):return json.dumps(clean(x),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csv_save(path,rows):
    rows=list(rows);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clean(rows))
def safe(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Caminho inválido: '+name)
    return p
def open_zip(path):
    z=zipfile.ZipFile(path)
    if len(z.namelist())!=len(set(z.namelist())):z.close();raise ValueError('ZIP com entradas repetidas.')
    for i in z.infolist():
        safe(i.filename)
        if stat.S_ISLNK(i.external_attr>>16):z.close();raise ValueError('Link simbólico no ZIP.')
    return z
def install_checked(raw,path,expected):
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Conteúdo não corresponde ao manifesto: '+str(path))
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.is_file() or sha(p)!=expected:p.write_bytes(raw)
    return p

class Store:
    def __init__(self,root):self.root=Path(root)
    def path(self,rel):safe(rel);return self.root/rel
    def fetch(self,rel,dest,expected=None):
        source=self.path(rel);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
        if not source.is_file():raise FileNotFoundError('Arquivo esperado no Drive: '+str(source))
        if dest.is_file() and expected and sha(dest)==expected:return dest
        t=dest.with_name(dest.name+'.part');shutil.copyfile(source,t)
        actual=sha(t)
        if expected and actual!=expected:t.unlink(missing_ok=True);raise ValueError('SHA-256 diferente: '+rel)
        t.replace(dest);return dest
    def put(self,source,rel,expected=None):
        source=Path(source);target=self.path(rel);target.parent.mkdir(parents=True,exist_ok=True)
        actual=sha(source);expected=expected or actual
        if actual!=expected:raise ValueError('Fonte do backup mudou: '+str(source))
        stamp=target.with_name(target.name+'.sha256.json')
        if target.is_file() and stamp.is_file():
            try:
                old=read(stamp)
                if old.get('sha256')==actual and target.stat().st_size==source.stat().st_size:return target
            except (OSError,ValueError):pass
        t=target.with_name(target.name+'.transferindo');shutil.copyfile(source,t)
        if t.stat().st_size!=source.stat().st_size:raise IOError('Backup incompleto: '+rel)
        t.replace(target);save(stamp,{'sha256':actual,'bytes':source.stat().st_size});return target

def download_weight(local):
    p=Path(local);p.parent.mkdir(parents=True,exist_ok=True)
    if p.is_file() and sha(p)==EXPECTED['rfdetr_sha256']:return p
    t=p.with_name(p.name+'.part');print('Baixando pré-treino oficial RF-DETR Small...',flush=True)
    with urllib.request.urlopen(EXPECTED['rfdetr_url'],timeout=60) as response,t.open('wb') as f:shutil.copyfileobj(response,f,4*1024**2)
    if sha(t,'md5')!=EXPECTED['rfdetr_md5'] or sha(t)!=EXPECTED['rfdetr_sha256']:t.unlink(missing_ok=True);raise ValueError('Pré-treino RF-DETR diferente da versão fixada.')
    t.replace(p);return p

def environment():
    import platform,torch
    names=['torch','torchvision','rfdetr','ultralytics','pytorch-lightning','transformers','supervision','numpy','pillow']
    return {'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in names},
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cuda':torch.version.cuda,
        'device':'cuda' if torch.cuda.is_available() else 'cpu'}

def package(output,name='pacote_experimento_24.zip'):
    output=Path(output)
    paths=[p for p in output.rglob('*') if p.is_file() and not any(x in p.parts for x in ['retomada','base','__pycache__'])
        and p.suffix not in {'.tmp','.part','.zip','.ckpt'} and not p.name.endswith('.transferindo')
        and not (p.name.startswith('checkpoint_') or p.name in {'last_ema.pth','modelo_final_regular.pth'})]
    paths=[p for p in paths if p.name!='integridade_pacote_24.json']
    save(output/'integridade_pacote_24.json',{'sha256':{p.relative_to(output).as_posix():sha(p) for p in paths}})
    paths.append(output/'integridade_pacote_24.json');target=output/name;t=target.with_suffix('.tmp')
    with zipfile.ZipFile(t,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(paths):z.write(p,p.relative_to(output).as_posix())
    t.replace(target);return target
