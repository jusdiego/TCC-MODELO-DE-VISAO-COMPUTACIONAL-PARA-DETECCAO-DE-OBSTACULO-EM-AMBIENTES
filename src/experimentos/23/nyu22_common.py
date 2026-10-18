"""Experimento compacto NYU: caminhos próprios, fontes e persistência verificável."""
from __future__ import annotations
from pathlib import Path,PurePosixPath
import csv,hashlib,json,math,os,shutil,time,urllib.request,zipfile
import numpy as np

NAMES=['bed','sofa','chair','table','door','window']
PT_NAMES=['Cama','Sofá','Cadeira','Mesa','Porta','Janela']
GLOBAL_IDS=[0,1,2,3,9,8]
BRANCH='TCC_Visao_Computacional/ramo_nyu22_23_v1'
SOURCES={
 'mat':{'url':'https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat',
        'sha256':'2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa','bytes':2972037809},
 'split':{'url':'https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat',
          'sha256':'6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee','bytes':2626},
 'weight':{'url':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
           'sha256':'1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a','bytes':22588772}}
DATA_SPEC={'version':'22-nyuv2-six-visible-instances-v1','images':1449,'official_train':795,'official_test':654,
    'width':640,'height':480,'seed':22002026,'val_fraction':.2,'split_trials':128,
    'min_train_images':300,'min_val_images':80,'min_train_gt_class':10,'min_val_gt_class':5,
    'phash_distance':4,'names':NAMES,'bounding_boxes':'extensão visível de cada par (rótulo original, instância), sem filtro de tamanho',
    'test_access_during_preparation':'conversão de formato e exclusão de duplicatas exatas; nenhuma previsão ou seleção usa TEST'}
TRAIN_SPEC={'version':'23-nyuv2-yolov8s-v1','epochs':80,'patience':15,'imgsz':640,'batch':8,'workers':2,
    'pretrained_name_aliases':{'couch':'sofa','dining table':'table'},
    'optimizer':'AdamW','lr0':.0005,'lrf':.01,'warmup_epochs':3.,'weight_decay':.0005,'cos_lr':True,'nbs':32,
    'amp':False,'mosaic':.5,'close_mosaic':10,'mixup':0.,'cutmix':0.,'copy_paste':0.,
    'degrees':0.,'translate':.1,'scale':.25,'fliplr':.5,'flipud':0.,'hsv_h':.015,'hsv_s':.4,'hsv_v':.3,
    'seed':23002026,'session_hours':4.,'inference_conf':.01,'nms_iou':.7,'match_iou':.5,'max_det':300,'predict_batch':8,
    'thresholds':[.05,.10,.15,.20,.25,.30,.40,.50,.60,.70,.80,.90],
    'target_precision':.8,'target_recall':.7,'bootstrap_groups':1000}

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,Path):return str(v)
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
def digest(value):return hashlib.sha256(json.dumps(clean(value),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csv_save(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def safe(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or '..' in p.parts or ':' in name:raise ValueError('Caminho inválido no pacote: '+name)
    return p
def extract(archive,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if len(set(z.namelist()))!=len(z.namelist()):raise ValueError('ZIP contém nomes repetidos.')
        for name in z.namelist():safe(name)
        z.extractall(root)
def zip_paths(target,root,paths):
    target=Path(target);root=Path(root);tmp=target.with_suffix('.tmp');target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(set(map(Path,paths))):
            name=path.relative_to(root).as_posix();safe(name)
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            with path.open('rb') as src,z.open(info,'w',force_zip64=True) as dst:shutil.copyfileobj(src,dst,2**20)
    tmp.replace(target);return target

def download(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file() and target.stat().st_size==source['bytes'] and sha(target)==source['sha256']:return target
    part=target.with_name(target.name+'.part');last=None
    for attempt in range(3):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset>=source['bytes']:part.unlink();offset=0
            request=urllib.request.Request(source['url'],headers={'User-Agent':'TCC-NYU-22/1.0',**({'Range':f'bytes={offset}-'} if offset else {})})
            with urllib.request.urlopen(request,timeout=60) as response:
                append=offset>0 and response.status==206
                if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Resposta HTTP de retomada incompatível.')
                if not append:offset=0
                done=offset;last_notice=0.
                with part.open('ab' if append else 'wb') as f:
                    while True:
                        block=response.read(4*1024**2)
                        if not block:break
                        f.write(block);done+=len(block)
                        if time.monotonic()-last_notice>30:
                            print(f'Download: {done/1024**2:.0f}/{source["bytes"]/1024**2:.0f} MiB',flush=True);last_notice=time.monotonic()
            if part.stat().st_size!=source['bytes']:raise IOError('Download incompleto.')
            if sha(part)!=source['sha256']:
                part.replace(part.with_name(part.name+'.rejeitado'));raise ValueError('SHA-256 diferente da fonte esperada.')
            part.replace(target);return target
        except Exception as exc:
            last=exc
            if attempt<2:time.sleep(2**attempt)
    raise RuntimeError('Não foi possível baixar e conferir '+source['url']+': '+str(last)) from last

class Storage:
    """Somente a pasta do ramo compacto; nunca sincroniza a raiz do TCC."""
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def path(self,relative):safe(relative);return self.root/relative
    def pull(self,relative,target,expected=None):
        source=self.path(relative);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
        receipt=source.with_name(source.name+'.sha256.json')
        if expected is None and receipt.is_file():expected=read(receipt)['sha256']
        if target.is_file() and expected and sha(target)==expected:return target
        if not source.is_file():raise FileNotFoundError('Arquivo ainda não disponível no ramo compacto: '+relative)
        tmp=target.with_name(target.name+'.part');shutil.copyfile(source,tmp)
        if expected and sha(tmp)!=expected:tmp.unlink(missing_ok=True);raise ValueError('Arquivo remoto incompleto ou alterado: '+relative)
        tmp.replace(target);return target
    def put(self,source,relative):
        source=Path(source);target=self.path(relative);target.parent.mkdir(parents=True,exist_ok=True)
        value=sha(source);receipt=target.with_name(target.name+'.sha256.json')
        if target.is_file() and receipt.is_file() and target.stat().st_size==source.stat().st_size:
            try:
                if read(receipt)['sha256']==value:return True
            except (ValueError,KeyError):pass
        tmp=target.with_name(target.name+'.tmp');shutil.copyfile(source,tmp)
        if sha(tmp)!=value:raise IOError('Falha ao conferir a cópia no Drive: '+relative)
        tmp.replace(target);save(receipt,{'sha256':value,'bytes':source.stat().st_size});return True
    def source(self,key,local,sources=SOURCES):
        spec=sources[key];name={'mat':'nyu_depth_v2_labeled.mat','split':'splits.mat','weight':'yolov8s.pt'}[key]
        local=Path(local)/name;remote='fontes/'+name
        if self.path(remote).is_file():return self.pull(remote,local,spec['sha256'])
        download(spec,local);self.put(local,remote);return local

def package(output,name):
    output=Path(output);paths=[];target=output/name
    for p in sorted(output.rglob('*')):
        if not p.is_file() or p==target or p.suffix in {'.tmp','.part','.zip'}:continue
        rel=p.relative_to(output)
        if '__pycache__' in rel.parts or 'retomada' in rel.parts or rel.as_posix().startswith('treino/weights/'):continue
        if p.suffix=='.pt' and rel.as_posix()!='melhor_modelo.pt':continue
        if p.name=='integridade.json':continue
        paths.append(p)
    save(output/'integridade.json',{'sha256':{p.relative_to(output).as_posix():sha(p) for p in paths}})
    return zip_paths(target,output,paths+[output/'integridade.json'])
