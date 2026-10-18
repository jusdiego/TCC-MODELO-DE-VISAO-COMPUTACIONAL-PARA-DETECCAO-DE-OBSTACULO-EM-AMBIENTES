"""Dataset comum: cópias com bytes e cobertura originais para os dois braços."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import shutil
import yaml
import diag11_core as core
import mix18_protocol as protocol

def make_common(records,home_val,destination,output):
    destination=Path(destination);output=Path(output);destination.mkdir(parents=True,exist_ok=True)
    if any(r['role']!='train' or r['source'] not in {'home','openimages'} for r in records):raise ValueError('Somente TRAIN entra no dataset comum.')
    if len({r['id'] for r in records})!=len(records):raise ValueError('Treino com IDs repetidos.')
    required=sum(Path(r['image']).stat().st_size for r in records)+256*1024**2
    if shutil.disk_usage(destination).free<required:raise RuntimeError('Espaço insuficiente para o treino comum.')
    mapping={};manifest=[];frozen=[]
    for r in protocol.logical_order(records):
        image=destination/'images/train'/protocol.filename(r)
        label=destination/'labels/train'/Path(protocol.filename(r)).with_suffix('.txt')
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        if core.sha256_file(r['image'])!=r['sha256'] or core.sha256_file(r['label'])!=r['label_sha256']:raise ValueError('Bytes de treino alterados antes da cópia.')
        if r['source']=='home' and r['known_mask']!=4095:raise ValueError('Home não pode receber máscara experimental.')
        shutil.copy2(r['image'],image);shutil.copy2(r['label'],label)
        if core.sha256_file(image)!=r['sha256'] or core.sha256_file(label)!=r['label_sha256']:raise ValueError('Cópia do treino divergente.')
        obj={**r,'image':image,'label':label}
        mapping[image.resolve().as_posix()]=obj
        manifest.append({k:r[k] for k in ['id','source','role','sha256','label_sha256','known_mask','width','height']})
        frozen.extend([(image,r['sha256']),(label,r['label_sha256'])])
    train=destination/'train.txt';train.write_text('\n'.join(sorted(mapping))+'\n')
    val=destination/'home_val.txt';val.write_text('\n'.join(str(Path(r['image']).resolve()) for r in home_val)+'\n')
    data=destination/'data.yaml';data.write_text(yaml.safe_dump({'path':str(destination.resolve()),'train':str(train.resolve()),'val':str(val.resolve()),'names':dict(enumerate(core.NAMES))},sort_keys=False,allow_unicode=True))
    frozen.extend([(f,core.sha256_file(f)) for f in [train,val,data]])
    core.save_json(output/'manifesto_treino_comum_18.json',manifest)
    return {a:{'records':mapping,'yaml':data} for a in protocol.ARMS},frozen
