"""Home histórico, sem leitura de imagens/rótulos TEST antes da decisão congelada."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import hashlib,json,shutil
import numpy as np
from PIL import Image
from scipy.fft import dctn
import yaml
import home24_common as c
from src.validacao_alertas import ler_objetos_yolo

QUARANTINE=['living_room_1p (200).jpg','living_room_1p (203).jpg','living_room_590.jpg','living_room_718.jpg']
PROTECTED=['living_room_1010.jpg','living_room_358.jpg']
SUFFIXES={'.jpg','.jpeg','.png','.bmp','.webp'}

def phash(image):
    with Image.open(image) as im:arr=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    vals=dctn(arr,type=2,norm='ortho')[:8,:8];bits=vals>np.median(vals[1:,:])
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(),'big')
def near(value,values,distance=4):return any((int(value)^int(x)).bit_count()<=distance for x in values)
def pixel_sha(path):
    with Image.open(path) as im:
        im=im.convert('RGB');return hashlib.sha256(c.canonical(im.size)+im.tobytes()).hexdigest()
def role_of(name):
    parts=Path(name).parts
    if 'images' not in parts:return None
    i=parts.index('images')
    if i+1>=len(parts) or parts[i+1] not in {'train','val','test'}:return None
    role=parts[i+1]
    # O ZIP histórico também pode conter SUN RGB-D. Reutiliza apenas o Home raw_train,
    # conforme os leitores 11/14/21; TEST é o Home histórico de 403 imagens.
    if role in {'train','val'} and (i+2>=len(parts) or parts[i+2]!='raw_train'):return None
    return role
def descriptor(name,entries):
    label=name.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
    if label not in entries:raise ValueError('Rótulo Home ausente: '+label)
    return {'id':'home:'+name,'member':name,'label_member':label,'role':role_of(name),
        'sha256':entries[name]['sha256'],'label_sha256':entries[label]['sha256'],'name':Path(name).name}
def materialise(z,desc,base):
    token=hashlib.sha256(desc['id'].encode()).hexdigest()[:24];role=desc['role']
    image=c.install_checked(z.read(desc['member']),Path(base)/'images'/role/(token+Path(desc['member']).suffix),desc['sha256'])
    label=c.install_checked(z.read(desc['label_member']),Path(base)/'labels'/role/(token+'.txt'),desc['label_sha256'])
    with Image.open(image) as im:w,h=im.size;im.verify()
    gt=ler_objetos_yolo(label,w,h,dict(enumerate(c.NAMES)))
    return {**desc,'image':image,'label':label,'width':w,'height':h,'gt':gt,'phash':phash(image),'pixel_sha256':pixel_sha(image)}
def serial(rows):return [{k:v for k,v in r.items() if k not in {'image','label'}} for r in rows]

def prepare(archive,base,output,expected=c.EXPECTED):
    archive=Path(archive);base=Path(base);output=Path(output);base.mkdir(parents=True,exist_ok=True)
    if c.sha(archive)!=expected['home_sha']:raise ValueError('O ZIP Home não é a versão histórica fixada.')
    with c.open_zip(archive) as z:
        m=json.loads(z.read('verificacao_preparacao.json'))
        if m['test']['imagens']!=403 or m['test']['sha256']!=expected['home_test_sha']:raise ValueError('TEST Home histórico diferente.')
        if {int(k):v for k,v in m['classes'].items()}!=dict(enumerate(c.NAMES)):raise ValueError('As 12 classes Home mudaram.')
        entries={r['arquivo']:r for r in m['arquivos']}
        descriptors=[descriptor(n,entries) for n in sorted(entries) if role_of(n) and Path(n).suffix.lower() in SUFFIXES]
        counts=Counter(r['role'] for r in descriptors)
        if dict(counts)!={'train':1824,'val':456,'test':403}:raise ValueError('Divisão Home inesperada: '+str(dict(counts)))
        reserve=[r for r in descriptors if r['role']=='test']
        val=[materialise(z,r,base) for r in descriptors if r['role']=='val']
        protected_sha={r['sha256'] for r in descriptors if r['role']!='train'}
        protected_ph=[r['phash'] for r in val];protected_pixel={r['pixel_sha256'] for r in val}
        train=[];excluded=[]
        for r in descriptors:
            if r['role']!='train':continue
            reason=('quarentena histórica de duplicatas' if r['name'] in QUARANTINE else
                    'exclusão protegida já documentada no 21' if r['name'] in PROTECTED else
                    'SHA idêntico ao VAL/TEST histórico' if r['sha256'] in protected_sha else None)
            if reason:excluded.append({'id':r['id'],'motivo':reason});continue
            row=materialise(z,r,base)
            if row['pixel_sha256'] in protected_pixel or near(row['phash'],protected_ph):
                excluded.append({'id':r['id'],'motivo':'pixels iguais ou pHash <=4 com VAL'});continue
            train.append(row)
        if len(train)<500 or len(val)!=456:raise ValueError('Base insuficiente após exclusões automáticas.')
    reference={'home_zip_sha256':expected['home_sha'],'classes':c.NAMES,'train':serial(train),'val':serial(val),
       'test_descriptors':reserve,'excluded':excluded,'test_previously_exposed':True,
       'test_read_for_training_or_selection':False,'phash_test_checked_before_training':False,
       'limite':'TEST possui hashes protegidos, mas não foi aberto para verificar semelhança perceptual nesta preparação.'}
    reference['dataset_id']=c.digest(reference);c.save(output/'manifesto_desenvolvimento_24.json',reference)
    c.save(output/'auditoria_base_24.json',{'train':len(train),'val':len(val),'test_historical':len(reserve),
       'classes':c.NAMES,'excluded':excluded,'test_images_or_labels_read':False,'all_12_classes_annotation_contract':True,
       'human_reannotation':False,'dataset_id':reference['dataset_id']})
    counts=[]
    for split,rows in [('train',train),('val',val)]:
        for cid,name in enumerate(c.NAMES):
            counts.append({'split':split,'class_id':cid,'class':name,'gt':sum(g['classe_id']==cid for r in rows for g in r['gt']),
              'images':sum(any(g['classe_id']==cid for g in r['gt']) for r in rows)})
    if any(r['gt']==0 for r in counts):raise ValueError('Classe sem suporte no desenvolvimento.')
    c.csv_save(output/'classes_desenvolvimento_24.csv',counts)
    config={'path':str(base.resolve()),'train':'images/train','val':'images/val','nc':12,'names':c.NAMES}
    (base/'data.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(f'Home preparado: {len(train)} TRAIN, {len(val)} VAL; {len(reserve)} TEST histórico protegido.',flush=True)
    return {'train':train,'val':val,'test_descriptors':reserve,'reference':reference,'base':base,'archive':archive}

def open_historical_test(job,decision,output):
    if not decision.get('frozen') or decision.get('dataset_id')!=job['reference']['dataset_id']:raise ValueError('Teste exige decisão congelada desta base.')
    if decision.get('decision_uses_test') is not False:raise ValueError('Decisão não autorizada para teste.')
    with c.open_zip(job['archive']) as z:rows=[materialise(z,r,job['base']) for r in job['test_descriptors']]
    if len(rows)!=403:raise ValueError('TEST histórico deve ter 403 imagens.')
    training=job['train'];pixels={r['pixel_sha256'] for r in training};hashes=[r['phash'] for r in training]
    overlap=[{'id':r['id'],'pixel_overlap':r['pixel_sha256'] in pixels,'phash_candidate':near(r['phash'],hashes)} for r in rows if r['pixel_sha256'] in pixels or near(r['phash'],hashes)]
    c.save(Path(output)/'auditoria_test_historico_24.json',{'images':403,'historically_exposed':True,
       'decision_key':c.digest(decision),'test_selection':False,'overlap_candidates':overlap,
       'nota':'Se houver coincidências, o resultado mantém as 403 imagens históricas, com exposição sinalizada; não é prova de generalização independente.'})
    return rows
