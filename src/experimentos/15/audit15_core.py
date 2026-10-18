"""Auditoria de TRAIN: identidade, caixas, cobertura e duplicações. Sem treinamento."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import stat
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from scipy.fft import dctn

VERSION = '15-auditoria-train-1.0.0'
NAMES = ['bed', 'sofa', 'chair', 'table', 'lamp', 'tv', 'laptop',
         'wardrobe', 'window', 'door', 'potted plant', 'photo frame']
ALL_KNOWN = (1 << len(NAMES)) - 1
FOCUS = {2, 3, 7, 10}


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def save_csv(path, rows, fields=None):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r)) or ['sem_registros']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def save_gz(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f, gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as g:
        g.write(canonical(obj).encode())
    tmp.replace(path)


def load_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def safe_member(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or '\\' in name
            or ':' in p.parts[0] or '\x00' in name):
        raise ValueError('Caminho inseguro no ZIP: ' + str(name))
    return p


def inspect_zip(z):
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError('ZIP com nomes duplicados.')
    for info in z.infolist():
        safe_member(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('Link simbólico não permitido no ZIP.')
    return set(names)


def checked_read(z, member, expected):
    safe_member(member)
    raw = z.read(member)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Conteúdo diferente do manifesto: ' + member)
    return raw


def phash_image(image):
    a = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(a, norm='ortho')[:8, :8]
    bits = low > np.median(low.flatten()[1:]); bits[0, 0] = False
    return int.from_bytes(np.packbits(bits.flatten()).tobytes(), 'big')


class NearIndex:
    def __init__(self):
        self.tables = [defaultdict(list) for _ in range(8)]

    def add(self, value, iid):
        for b in range(8):
            self.tables[b][(value >> (8*b)) & 255].append((value, iid))

    def find(self, value, max_distance=4, limit=5):
        found, seen = [], set()
        for b in range(8):
            for prior, iid in self.tables[b].get((value >> (8*b)) & 255, []):
                if iid in seen: continue
                seen.add(iid); distance = (prior ^ value).bit_count()
                if distance <= max_distance: found.append((distance, iid))
        return sorted(found)[:limit]


def iou_matrix(a, b):
    if not a or not b: return np.zeros((len(a), len(b)))
    a = np.asarray([r['bbox'] for r in a], float); b = np.asarray([r['bbox'] for r in b], float)
    low = np.maximum(a[:, None, :2], b[None, :, :2]); high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(high-low, 0).prod(2)
    area_a = np.maximum(a[:, 2:]-a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def parse_labels(raw, width, height):
    """Registra todas as linhas; nenhuma correção silenciosa dos rótulos."""
    boxes, issues = [], []
    try: text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        return [], [{'line': 0, 'code': 'label_nao_utf8', 'blocking': True}]
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        base = {'line': ln, 'raw': line[:180]}
        try:
            vals = [float(x) for x in line.split()]
            if len(vals) != 5 or not all(math.isfinite(x) for x in vals): raise ValueError()
            cid, x, y, w, h = vals
            if not cid.is_integer() or not 0 <= cid < len(NAMES): raise ValueError()
        except ValueError:
            issues.append({**base, 'code': 'linha_ou_classe_invalida', 'blocking': True}); continue
        norm = [x-w/2, y-h/2, x+w/2, y+h/2]
        if min(w, h) <= 0 or min(norm) < -1e-6 or max(norm) > 1+1e-6:
            issues.append({**base, 'code': 'caixa_fora_ou_area_invalida', 'blocking': True}); continue
        box = [norm[0]*width, norm[1]*height, norm[2]*width, norm[3]*height]
        side = min(w*width, h*height)*640/max(width, height)
        flags = []
        if side < 16: flags.append('menor_lado_em_640_inferior_a_16')
        if max(w*width/(h*height), h*height/(w*width)) > 12: flags.append('proporcao_extrema_maior_12')
        if min(norm) < 0 or max(norm) > 1: flags.append('arredondamento_na_borda_ate_1e6')
        obj = {'index': len(boxes), 'line': ln, 'classe_id': int(cid), 'classe': NAMES[int(cid)],
               'bbox_norm': norm, 'bbox': box, 'area_relativa': w*h,
               'menor_lado_em_640': side, 'flags': flags}
        boxes.append(obj)
        for flag in flags: issues.append({**base, 'code': flag, 'blocking': False})
    matrix = iou_matrix(boxes, boxes)
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            same = boxes[i]['classe_id'] == boxes[j]['classe_id']
            if matrix[i, j] >= (.95 if same else .90):
                code = 'caixas_mesma_classe_quase_duplicadas' if same else 'classes_diferentes_sobreposicao_alta'
                issues.append({'line': boxes[i]['line'], 'other_line': boxes[j]['line'],
                               'code': code, 'iou': float(matrix[i,j]), 'blocking': False})
    return boxes, issues


def validate_coverage(cov, known, gt, source):
    if source == 'home': return []
    problems = []
    pos = cov.get('classes_positivas_verificadas_exaustivas', [])
    neg = cov.get('classes_negativas_verificadas', [])
    ids = pos + neg
    if (any(type(c) is not int or not 0 <= c < 12 for c in ids)
            or type(known) is not int or not 0 <= known <= ALL_KNOWN):
        return [{'line': 0, 'code': 'mascara_cobertura_invalida', 'blocking': True}]
    if (set(pos) & set(neg) or sum(1 << c for c in set(ids)) != known
            or cov.get('known_mask') != known):
        problems.append({'line': 0, 'code': 'cobertura_inconsistente', 'blocking': True})
    present = {o['classe_id'] for o in gt}
    if present & set(neg): problems.append({'line': 0, 'code': 'caixa_em_classe_negativa_verificada', 'blocking': True})
    if set(pos) - present: problems.append({'line': 0, 'code': 'classe_positiva_exaustiva_sem_caixa', 'blocking': True})
    return problems


def load_history(csv_path, json_path, expected):
    split = json.loads(Path(json_path).read_text()); claimed = split.pop('divisao_sha256')
    if claimed != expected or digest(split) != claimed: raise ValueError('Divisão histórica alterada.')
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    if (len(rows) != 100000 or len({r['id'] for r in rows}) != 100000
            or Counter(r['split'] for r in rows) != {'train': 80000, 'test': 20000}):
        raise ValueError('Divisão diferente dos 100 mil IDs históricos.')
    for role, key in [('train', 'ids_treino'), ('test', 'ids_teste')]:
        if {r['id'] for r in rows if r['split'] == role} != set(split[key]):
            raise ValueError('CSV/JSON históricos divergem.')
    if len({r['imagem_sha256'] for r in rows}) != len(rows): raise ValueError('Histórico com bytes repetidos.')
    return rows


def history_distribution(rows, sample_ids):
    result = []
    for label, cohort in [('train_historico_80000', [r for r in rows if r['split']=='train']),
                          ('recorte_cinco_lotes', [r for r in rows if r['id'] in sample_ids])]:
        for cid, name in enumerate(NAMES):
            result.append({'coorte': label, 'classe_id': cid, 'classe': name, 'imagens': len(cohort),
                           'classe_primaria_imagens': sum(int(r['classe_primaria']) == cid for r in cohort),
                           'classe_com_cobertura_imagens': sum(bool(int(r['known_mask']) & (1 << cid)) for r in cohort)})
    return result


def home_catalog(zip_path, expected):
    if sha_file(zip_path) != expected['home_zip']: raise ValueError('ZIP Home alterado.')
    with zipfile.ZipFile(zip_path) as z:
        inspect_zip(z); manifest = json.loads(z.read('verificacao_preparacao.json'))
    if (manifest['test']['sha256'] != expected['test_home'] or manifest['test']['imagens'] != 403
            or {int(k): v for k,v in manifest['classes'].items()} != dict(enumerate(NAMES))):
        raise ValueError('Manifesto Home incompatível.')
    root = manifest['dataset_dir']; items = {r['arquivo']: r for r in manifest['arquivos']}
    if len(items) != len(manifest['arquivos']): raise ValueError('Manifesto Home repetido.')
    images = {name: r for name,r in items.items() if name.startswith(root+'/images/')
              and Path(name).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}}
    train = {n:r for n,r in images.items() if n.startswith(root+'/images/train/raw_train/')}
    val = {n:r for n,r in images.items() if n.startswith(root+'/images/val/raw_train/')}
    if len(train)!=1824 or len(val)!=456: raise ValueError('Quantidades Home incompatíveis.')
    protected = [(r['sha256'], 'home_protegido:'+n) for n,r in images.items() if n not in train]
    catalog = []
    for name,item in sorted(train.items()):
        label = name.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
        if label not in items: raise ValueError('Label Home ausente do manifesto.')
        catalog.append({'id':'home:'+name, 'source':'home', 'role':'train', 'image_member':name,
                        'label_member':label, 'sha256':item['sha256'], 'label_sha256':items[label]['sha256'],
                        'known_mask':ALL_KNOWN, 'lote':'homeobjects_sunrgbd_validado.zip'})
    return catalog, protected


def protected_index(history, home_protected, new_manifest):
    by_sha, ids, near = defaultdict(list), set(), NearIndex()
    for sha, iid in home_protected: by_sha[sha].append(iid)
    for r in history:
        if r['split']=='test': by_sha[r['imagem_sha256']].append('oi_test:'+r['id']); ids.add(r['id'])
    for r in new_manifest['arquivos']:
        by_sha[r['sha256']].append(r['role']+':'+r['id']); ids.add(r['id'])
        if r.get('phash') is not None:
            near.add(int(r['phash'],16), r['role']+':'+r['id'])
    return by_sha, ids, near


def materialize_one(z, spec, output, protected_sha, protected_ids, protected_near):
    if spec['role'] != 'train': raise ValueError('Papel proibido na auditoria de treino.')
    iid = spec.get('oi_id', spec['id'])
    hit = protected_sha.get(spec['sha256'], [])
    record = {**spec, 'gt': [], 'issues': [], 'protected_matches': list(hit), 'image': None}
    if hit or iid in protected_ids:
        record['issues'].append({'line':0, 'code':'coincidencia_com_conjunto_protegido', 'blocking':True})
        record['status_input']='quarentena_sem_extracao'; return record
    raw_label = checked_read(z, spec['label_member'], spec['label_sha256'])
    cov = {}
    if spec.get('coverage_member'):
        cov = json.loads(checked_read(z, spec['coverage_member'], spec['coverage_sha256']))
    raw = checked_read(z, spec['image_member'], spec['sha256'])
    record['label_text'] = raw_label.decode('utf-8-sig', errors='replace'); record['coverage_original']=cov
    try:
        with Image.open(io.BytesIO(raw)) as im:
            width,height = im.size; im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            im.load(); ph = phash_image(im)
            orientation = im.getexif().get(274,1)
        if spec.get('width') is not None and (spec['width'],spec['height']) != (width,height):
            record['issues'].append({'line':0,'code':'dimensoes_divergem_do_manifesto','blocking':True})
        record.update(width=width,height=height,phash=f'{ph:016x}',exif_orientation=int(orientation))
        hits = protected_near.find(ph)
        record['protected_phash_matches'] = [{'distancia':d,'id':n} for d,n in hits]
        if hits: record['issues'].append({'line':0,'code':'semelhanca_phash_com_protegido','blocking':False})
        if orientation != 1:
            record['issues'].append({'line':0,'code':'orientacao_exif_nao_padrao','blocking':True})
        gt, issues = parse_labels(raw_label,width,height)
        record['gt']=gt; record['issues'] += issues + validate_coverage(cov,spec['known_mask'],gt,spec['source'])
        if not gt: record['issues'].append({'line':0,'code':'imagem_sem_caixas','blocking':False})
        # Não inferir pixels de uma possível cópia perceptual da reserva/validação.
        if hits: record['status_input']='quarentena_phash_sem_inferencia'; return record
        image = Path(output)/'images'/spec['source']/(hashlib.sha256(spec['id'].encode()).hexdigest()+'.jpg')
        image.parent.mkdir(parents=True,exist_ok=True); image.write_bytes(raw)
        record['image']=str(image)
        record['status_input']='bloqueio_tecnico' if any(i['blocking'] for i in record['issues']) else 'legivel'
    except (OSError,ValueError,Image.DecompressionBombError) as exc:
        record['issues'].append({'line':0,'code':'imagem_ilegivel','blocking':True,'detail':str(exc)[:160]})
        record['status_input']='bloqueio_tecnico'
    return record


def extract_home(zip_path,catalog,output,protected,progress=None):
    records=[]
    with zipfile.ZipFile(zip_path) as z:
        inspect_zip(z)
        for i,spec in enumerate(catalog,1):
            records.append(materialize_one(z,spec,output,*protected))
            if progress and (i%200==0 or i==len(catalog)): progress(f'Home: {i}/{len(catalog)} imagens auditadas')
    return records


def extract_oi_package(zip_path,item,history_map,expected_cfg,output,protected):
    if sha_file(zip_path)!=item['zip_sha256'] or Path(zip_path).stat().st_size!=item['zip_bytes']:
        raise ValueError('Lote incompleto ou diferente do histórico: '+item['arquivo'])
    records=[]
    with zipfile.ZipFile(zip_path) as z:
        inspect_zip(z); m=json.loads(z.read('manifesto_lote.json'))
        if m['config_sha256']!=expected_cfg or m['origem_split']!='train': raise ValueError('Lote fora do TRAIN esperado.')
        accepted=m['aceitas']
        expected_ids={r['id'] for r in history_map.values() if r['lote']==item['arquivo']}
        if len(accepted)!=len(expected_ids) or {r['id'] for r in accepted}!=expected_ids:
            raise ValueError('IDs do lote divergentes da divisão histórica.')
        for r in sorted(accepted,key=lambda r:r['id']):
            h=history_map[r['id']]
            if (h['split']!='train' or h['lote']!=item['arquivo'] or h['imagem_sha256']!=r['sha256']
                    or int(h['known_mask'])!=r['known_mask']): raise ValueError('Identidade de treino incompatível.')
            spec={'id':'oi:'+r['id'],'oi_id':r['id'],'source':'openimages','role':'train',
                  'image_member':r['imagem'],'label_member':r['label'],'coverage_member':r['cobertura'],
                  'sha256':r['sha256'],'label_sha256':r['label_sha256'],'coverage_sha256':r['cobertura_sha256'],
                  'known_mask':r['known_mask'],'width':r['width'],'height':r['height'],
                  'lote':item['arquivo'],'classe_primaria':int(h['classe_primaria'])}
            records.append(materialize_one(z,spec,output,*protected))
    return records


def duplicate_rows(records):
    exact, rows, near = defaultdict(list), [], NearIndex()
    for r in sorted(records,key=lambda r:r['id']):
        for prior in exact[r['sha256']]:
            rows.append({'id':r['id'],'similar_id':prior,'type':'bytes_iguais','distance':0})
        exact[r['sha256']].append(r['id'])
        if r.get('phash') is not None:
            value=int(r['phash'],16)
            for distance,prior in near.find(value):
                rows.append({'id':r['id'],'similar_id':prior,'type':'semelhanca_phash','distance':distance})
            near.add(value,r['id'])
    return rows


def manifest_public(records):
    return [{k:v for k,v in r.items() if k!='image'} for r in records]


def assert_inputs_unchanged(records):
    for r in records:
        if r.get('image') and sha_file(r['image'])!=r['sha256']:
            raise ValueError('Imagem de treino mudou: '+r['id'])


def eligible(records):
    return [r for r in records if r['role']=='train' and r['status_input']=='legivel' and r.get('image')]


def chunks(records,n=100):
    for source in sorted({r['source'] for r in records}):
        group=sorted([r for r in records if r['source']==source],key=lambda r:r['id'])
        for i in range(0,len(group),n): yield source+f'_{i//n:04d}',group[i:i+n]


def record_identity(r):
    return {k:r.get(k) for k in ['id','role','source','sha256','label_sha256','coverage_sha256','known_mask','width','height']}


def cache_read(path,signature,ids,validator=None):
    path=Path(path)
    if not path.is_file(): return None
    obj=load_gz(path)
    if obj.get('signature')!=signature or [r['id'] for r in obj.get('rows',[])]!=ids:
        raise ValueError('Cache incompatível: '+str(path))
    if obj.get('rows_sha256')!=digest(obj['rows']):
        raise ValueError('Conteúdo do cache alterado: '+str(path))
    if validator:
        for r in obj['rows']: validator(r)
    return obj['rows']


def cache_write(path,signature,rows):
    save_gz(path,{'signature':signature,'rows_sha256':digest(rows),'rows':rows})


def recoverable_cache_read(path,signature,ids,validator=None):
    try:return cache_read(path,signature,ids,validator)
    except (ValueError,KeyError,TypeError,EOFError,OSError) as exc:
        path=Path(path)
        rejected=path.parent/'rejeitados';rejected.mkdir(exist_ok=True)
        target=rejected/(path.name+'.'+str(time.time_ns())+'.rejeitado')
        if path.exists():path.replace(target)
        save_json(str(target)+'.json',{'cache':path.name,'motivo':str(exc)[:240],'recalculado_automaticamente':True})
        print('Cache rejeitado; recalculando automaticamente:',path.name,flush=True)
        return None


def package(output):
    output=Path(output); archive=output/'pacote_auditoria_15.zip'; tmp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p not in {archive,tmp} and not p.name.endswith('.tmp'):
                z.write(p,p.relative_to(output))
    tmp.replace(archive)
    return archive
