"""Máscaras humanas NYUv2 -> caixas visíveis; separação por cenas e pHash."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,re
import h5py,numpy as np,yaml
from PIL import Image,ImageDraw
from scipy.fft import dctn
from scipy.io import loadmat
import nyu22_common as c

def strings(f,key):return [''.join(chr(int(v)) for v in f[ref][:].reshape(-1)) for ref in f[key][:].reshape(-1)]
def scene_group(name):return re.sub(r'(?<=\d)[a-z]+$','',name.strip().lower())
def phash(rgb):
    gray=np.asarray(Image.fromarray(rgb).convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    low=dctn(gray,norm='ortho')[:8,:8];bits=low>np.median(low.ravel()[1:]);bits[0,0]=False
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(),'big')

def boxes_from_masks(labels,instances,name_to_id):
    if labels.shape!=instances.shape or labels.ndim!=2:raise ValueError('Máscaras incompatíveis.')
    result=[]
    for cid,name in enumerate(c.NAMES):
        raw_id=name_to_id[name];selected=labels==raw_id
        for instance in np.unique(instances[selected]):
            # Mesmo critério do get_instance_masks.m oficial: rótulo + instância.
            yy,xx=np.where(selected&(instances==instance))
            if not len(xx):continue
            result.append({'class_id':cid,'class':name,'bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                'mask_pixels':int(len(xx)),'raw_label':int(raw_id),'instance':int(instance)})
    return result
def yolo_bytes(boxes,w,h):
    lines=[]
    for obj in boxes:
        x1,y1,x2,y2=obj['bbox']
        if not(0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Caixa derivada inválida.')
        lines.append(f'{obj["class_id"]} {(x1+x2)/(2*w):.17g} {(y1+y2)/(2*h):.17g} {(x2-x1)/w:.17g} {(y2-y1)/h:.17g}\n')
    return ''.join(lines).encode()

def convert(f,index,role,root,name_to_id,scenes,spec):
    i=index-1;rgb=np.asarray(f['images'][i]).transpose(2,1,0)
    labels=np.asarray(f['labels'][i]).T;instances=np.asarray(f['instances'][i]).T
    if rgb.shape!=(spec['height'],spec['width'],3) or labels.shape!=rgb.shape[:2]:raise ValueError('Orientação ou dimensão NYU inesperada.')
    boxes=boxes_from_masks(labels,instances,name_to_id);iid=f'nyu_{index:05d}'
    image=Path(root)/'images'/role/(iid+'.png');label=Path(root)/'labels'/role/(iid+'.txt')
    image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(image);label.write_bytes(yolo_bytes(boxes,spec['width'],spec['height']))
    return {'id':iid,'index':index,'role':role,'scene':scenes[i],'scene_group':scene_group(scenes[i]),
        'image':image.relative_to(root).as_posix(),'label':label.relative_to(root).as_posix(),
        'image_sha256':c.sha(image),'label_sha256':c.sha(label),'pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
        'width':spec['width'],'height':spec['height'],'phash':phash(rgb),'gt':boxes,
        'unlabeled_fraction':float((labels==0).mean())}

def groups_for(rows,spec):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    scenes={};bands=[defaultdict(list) for _ in range(8)];similar=[]
    for i,r in enumerate(rows):
        if r['scene_group'] in scenes:union(i,scenes[r['scene_group']])
        scenes.setdefault(r['scene_group'],i)
        candidates={j for k in range(8) for j in bands[k][(r['phash']>>(8*k))&255]}
        for j in candidates:
            if (r['phash']^rows[j]['phash']).bit_count()<=spec['phash_distance']:
                union(i,j)
                if r['scene_group']!=rows[j]['scene_group']:similar.append([r['id'],rows[j]['id']])
        for k in range(8):bands[k][(r['phash']>>(8*k))&255].append(i)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    for group in groups.values():
        key=min(r['id'] for r in group)
        for r in group:r['split_group']=key
    return sorted(groups.values(),key=lambda g:min(r['id'] for r in g)),similar

def split_development(rows,spec):
    groups,similar=groups_for(rows,spec);total=np.bincount([o['class_id'] for r in rows for o in r['gt']],minlength=6)
    options=[];n=max(1,round(len(groups)*spec['val_fraction']))
    for trial in range(spec['split_trials']):
        order=np.random.default_rng(spec['seed']+trial).permutation(len(groups));v={int(x) for x in order[:n]}
        val=[r for i,g in enumerate(groups) if i in v for r in g];train=[r for i,g in enumerate(groups) if i not in v for r in g]
        counts=np.bincount([o['class_id'] for r in val for o in r['gt']],minlength=6)
        if len(train)<spec['min_train_images'] or len(val)<spec['min_val_images']:continue
        if (counts<spec['min_val_gt_class']).any() or ((total-counts)<spec['min_train_gt_class']).any():continue
        score=abs(len(val)/len(rows)-spec['val_fraction'])+float(np.mean(abs(counts/np.maximum(total,1)-spec['val_fraction'])))
        options.append((score,trial,train,val))
    if not options:raise ValueError('Não foi possível obter TRAIN/VAL com cenas separadas e suporte mínimo nas seis classes.')
    _,trial,train,val=min(options,key=lambda x:(x[0],x[1]))
    for role,group in [('train',train),('val',val)]:
        for r in group:r['role']=role
    return sorted(train,key=lambda r:r['id']),sorted(val,key=lambda r:r['id']),{'trial':trial,'groups':len(groups),'phash_pairs_across_scenes':similar,
        'criterion':'Proporções de imagens e caixas, antes de qualquer treinamento; somente TRAIN oficial. Não usa precisão, recall ou rótulos de TEST.'}

def relocate(rows,root):
    for r in rows:
        for key,folder in [('image','images'),('label','labels')]:
            old=Path(root)/r[key];new=Path(root)/folder/r['role']/old.name
            if old!=new:new.parent.mkdir(parents=True,exist_ok=True);old.replace(new);r[key]=new.relative_to(root).as_posix()
def contact_sheet(rows,root,target):
    selected=rows[:12];page=Image.new('RGB',(900,math_ceil(len(selected)/3)*250),'white');draw=ImageDraw.Draw(page)
    for i,r in enumerate(selected):
        im=Image.open(Path(root)/r['image']).convert('RGB');d=ImageDraw.Draw(im)
        for o in r['gt']:
            d.rectangle(o['bbox'],outline='lime',width=2);d.text(o['bbox'][:2],c.PT_NAMES[o['class_id']],fill='yellow')
        im.thumbnail((294,220));x=(i%3)*300;y=(i//3)*250;page.paste(im,(x,y));draw.text((x,y+222),r['id']+' | '+r['role'],fill='black')
    page.save(target)
def math_ceil(v):return int(np.ceil(v))

def prepare(mat,split,root,output,spec=c.DATA_SPEC):
    root=Path(root);output=Path(output);root.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    split=loadmat(split);train_ids=[int(x) for x in split['trainNdxs'].reshape(-1)];test_ids=[int(x) for x in split['testNdxs'].reshape(-1)]
    if len(train_ids)!=spec['official_train'] or len(test_ids)!=spec['official_test'] or set(train_ids)&set(test_ids) or set(train_ids+test_ids)!=set(range(1,spec['images']+1)):
        raise ValueError('Divisão oficial incompleta ou sobreposta.')
    with h5py.File(mat,'r') as f:
        names=strings(f,'names');scenes=strings(f,'scenes')
        mapping={n:names.index(n)+1 for n in c.NAMES}
        if len(scenes)!=spec['images'] or f['images'].shape!=(spec['images'],3,spec['width'],spec['height']):raise ValueError('Esquema NYU diferente da fonte esperada.')
        train_scenes={scene_group(scenes[i-1]) for i in train_ids};test_scenes={scene_group(scenes[i-1]) for i in test_ids}
        if train_scenes&test_scenes:raise ValueError('Cenas da reserva aparecem no TRAIN oficial.')
        pool=[]
        for k,index in enumerate(train_ids,1):
            pool.append(convert(f,index,'pool',root,mapping,scenes,spec))
            if k%100==0:print(f'Conversão do TRAIN oficial: {k}/{len(train_ids)}',flush=True)
        train,val,allocation=split_development(pool,spec);relocate(train+val,root)
        reserve=[];pixels={r['pixel_sha256'] for r in pool}
        for k,index in enumerate(test_ids,1):
            r=convert(f,index,'test',root,mapping,scenes,spec)
            if r['pixel_sha256'] in pixels:raise ValueError('Imagem da reserva coincide exatamente com o desenvolvimento.')
            reserve.append(r)
            if k%100==0:print(f'Conversão da reserva, sem avaliação: {k}/{len(test_ids)}',flush=True)
    if {r['split_group'] for r in train}&{r['split_group'] for r in val}:raise ValueError('Grupo repetido entre TRAIN e VAL.')
    counts=[{'split':role,'class':name,'gt':sum(o['class_id']==cid for r in rows for o in r['gt']),
             'images':sum(any(o['class_id']==cid for o in r['gt']) for r in rows)}
            for role,rows in [('train',train),('val',val)] for cid,name in enumerate(c.NAMES)]
    report={'version':spec['version'],'train_images':len(train),'val_images':len(val),'test_images_reserved':len(reserve),
        'train_scenes':len({r['scene_group'] for r in train}),'val_scenes':len({r['scene_group'] for r in val}),'test_scenes':len(test_scenes),
        'scene_overlap_train_val':0,'scene_overlap_development_test':0,'class_counts_development':counts,'allocation':allocation,
        'raw_label_ids':mapping,'unlabeled_pixel_fraction_development':float(np.mean([r['unlabeled_fraction'] for r in pool])),
        'classes':c.NAMES,'global_ids_old_project':c.GLOBAL_IDS,'manual_labels_required':False,'test_used_for_selection':False,
        'task':'Detecção 2D em seis categorias NYU; caixas derivadas das máscaras visíveis, sem profundidade como entrada.',
        'limits':['Escopo e referência distintos do experimento 21; percentuais não são diretamente comparáveis.',
                  'SUN RGB-D incorpora imagens NYU: não reutilizar pesos Home/C10/21 nem declarar comparação independente com eles.',
                  'Pixels sem rótulo existem na origem. As caixas representam os objetos anotados, não uma garantia de anotação perfeita.']}
    manifest={'version':spec['version'],'classes':c.NAMES,'train':train,'val':val,'test_ids':test_ids,'test_scenes':sorted(test_scenes),'report':report}
    c.save(root/'manifesto_desenvolvimento.json',manifest);c.save(root/'manifesto_reserva.json',{'classes':c.NAMES,'test':reserve})
    c.save(output/'preparacao_nyu22.json',report);c.csv_save(output/'classes_desenvolvimento.csv',counts)
    contact_sheet(train[:6]+val[:6],root,output/'exemplos_desenvolvimento.jpg')
    dev_paths=[root/'manifesto_desenvolvimento.json']+[root/r[k] for r in train+val for k in ['image','label']]
    test_paths=[root/'manifesto_reserva.json']+[root/r[k] for r in reserve for k in ['image','label']]
    dev=c.zip_paths(output/'desenvolvimento_nyu22.zip',root,dev_paths);test=c.zip_paths(output/'reserva_nyu22.zip',root,test_paths)
    return report,dev,test

def load_development(archive,root):
    root=Path(root);c.extract(archive,root);m=c.read(root/'manifesto_desenvolvimento.json')
    if m['classes']!=c.NAMES:raise ValueError('Classes da base preparada diferentes.')
    train=m['train'];val=m['val']
    if any(r['role']!='train' for r in train) or any(r['role']!='val' for r in val):raise ValueError('Papéis trocados na base.')
    if {r['index'] for r in train+val}&set(m['test_ids']):raise ValueError('TEST entrou no desenvolvimento.')
    if {r['split_group'] for r in train}&{r['split_group'] for r in val}:raise ValueError('Grupos sobrepostos.')
    hydrate(train+val,root)
    yaml_file=write_yaml(train,val,root,'data.yaml')
    return train,val,yaml_file,m
def hydrate(rows,root):
    for r in rows:
        for key,hkey in [('image','image_sha256'),('label','label_sha256')]:
            c.safe(r[key]);p=Path(root)/r[key]
            if c.sha(p)!=r[hkey]:raise ValueError('Imagem ou rótulo preparado foi alterado: '+r['id'])
            r[key]=p
def write_yaml(train,val,root,name):
    root=Path(root);a=root/(name+'.train.txt');b=root/(name+'.val.txt')
    a.write_text('\n'.join(str(Path(r['image']).resolve()) for r in train)+'\n');b.write_text('\n'.join(str(Path(r['image']).resolve()) for r in val)+'\n')
    target=root/name;target.write_text(yaml.safe_dump({'path':str(root.resolve()),'train':str(a.resolve()),'val':str(b.resolve()),'names':dict(enumerate(c.NAMES))},sort_keys=False));return target
def load_reserve(archive,root,development_manifest):
    root=Path(root);c.extract(archive,root);m=c.read(root/'manifesto_reserva.json');rows=m['test']
    if m['classes']!=c.NAMES or any(r['role']!='test' for r in rows):raise ValueError('Reserva incompatível.')
    if {r['index'] for r in rows}!=set(development_manifest['test_ids']):raise ValueError('IDs de TEST foram substituídos.')
    protected=development_manifest['train']+development_manifest['val']
    if {r['pixel_sha256'] for r in rows}&{r['pixel_sha256'] for r in protected}:raise ValueError('Pixels repetidos entre desenvolvimento e reserva.')
    if {r['scene_group'] for r in rows}&{r['scene_group'] for r in protected}:raise ValueError('Cenas da reserva sobrepostas.')
    hydrate(rows,root);return rows
