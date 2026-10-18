"""Protocolo pareado: somente a supervisão Home/wardrobe muda entre braços."""
from __future__ import annotations
import gzip,hashlib,json,math
from pathlib import Path
from collections import Counter
import exp12_protocol as historical

VERSION='16-home-unknown7-pareado-1.0.0'
ARMS=('C_original','M_home_unknown7')
WARDROBE=7
SPEC={'epochs':3,'batches':114,'batch':16,'oi_per_batch':4,'lr0':3e-6,
      'home_unique':1820,'oi_unique':2000,'seed':42}
AUGMENTATION_ZERO=['mosaic','mixup','cutmix','copy_paste','scale','translate','fliplr','flipud',
                   'degrees','shear','perspective','hsv_h','hsv_s','hsv_v','multi_scale']

def normalize(x):
    if isinstance(x,dict):return {str(k):normalize(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [normalize(v) for v in x]
    return x
def canonical(x):return json.dumps(normalize(x),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def save_gz(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(canonical(obj).encode())
    tmp.replace(path)
def load_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

def filename(record):
    prefix='home' if record['source']=='home' else 'oi'
    return prefix+'_'+hashlib.sha256(record['id'].encode()).hexdigest()+'.jpg'

def project_label(raw,source,arm,known):
    if arm not in ARMS or source not in {'home','openimages'}:raise ValueError('Braço/fonte desconhecido.')
    if not 0<=known<4096:raise ValueError('Máscara inválida.')
    lines=raw.decode('utf-8-sig').splitlines(keepends=True)
    removed=[];kept=[]
    for line in lines:
        if not line.strip():kept.append(line);continue
        parts=line.split()
        if len(parts)!=5:raise ValueError('Label deve conter cinco campos por linha.')
        values=[float(v) for v in parts]
        if not all(map(math.isfinite,values)) or not values[0].is_integer() or not 0<=values[0]<12:
            raise ValueError('Classe ou valor inválido no label.')
        cid=int(values[0])
        if arm==ARMS[1] and source=='home' and cid==WARDROBE:removed.append(line)
        else:kept.append(line)
    if arm==ARMS[1] and source=='home':
        return ''.join(kept).encode('utf-8'),known&~(1<<WARDROBE),len(removed)
    return raw,known,0

def logical_order(records):return sorted(records,key=filename)

def schedule(records,spec=SPEC):
    ordered=logical_order(records)
    home=[i for i,r in enumerate(ordered) if r['source']=='home']
    oi=[i for i,r in enumerate(ordered) if r['source']=='openimages']
    sampler=historical.BudgetSampler(home,oi,spec['batch'],spec['batches'],spec['oi_per_batch'],spec['seed'])
    rows=[]
    for epoch in range(spec['epochs']):
        sampler.set_epoch(epoch);seq=list(sampler)
        for offset in range(0,len(seq),spec['batch']):
            chosen=[ordered[i] for i in seq[offset:offset+spec['batch']]]
            ids=[r['id'] for r in chosen]
            rows.append({'step':len(rows)+1,'epoch':epoch+1,'ids':ids,
                         'input_signature':digest([(r['id'],r['sha256']) for r in chosen])})
    return rows

def same_other_classes(raw_a,raw_b):
    def rows(raw):
        return [tuple(map(float,l.split())) for l in raw.decode('utf-8-sig').splitlines() if l.strip() and int(float(l.split()[0]))!=WARDROBE]
    return rows(raw_a)==rows(raw_b)

def safe_judge(home,external,rule):
    valid=all(s['alertas_por_objeto'][k] is not None and math.isfinite(s['alertas_por_objeto'][k])
              for s in [home,external] for k in ['precision','recall','f1'])
    if valid:return {**historical.judge(home,external,rule),'metricas_com_suporte':True}
    return {'elegivel':False,'retencao_externa':False,'metricas_com_suporte':False}

def select_final(entries,rule):
    assessed=[{**e,'checks':safe_judge(e['home'],e['external'],rule)} for e in entries]
    primary=[e for e in assessed if e['arm'] in ARMS and e['epoch']==SPEC['epochs']]
    if len(primary)!=2:raise ValueError('Exigidos os dois checkpoints finais pré-fixados.')
    good=[e for e in primary if e['checks']['elegivel']]
    def key(e):
        h=e['home']['alertas_por_objeto'];x=e['external']['alertas_por_objeto']
        return (-h['precision'],-h['recall'],e['home']['alta_sem_alerta'],-x['f1'],e['id'])
    chosen=min(good,key=key) if good else None
    return {'assessed':assessed,'primary_ids':[e['id'] for e in primary],
            'eligible':chosen,'status':'checkpoint_final_elegivel_em_validacao' if chosen else 'nenhum_checkpoint_final_elegivel',
            'checkpoints_intermediarios':'somente diagnóstico, sem substituir o desfecho principal e03',
            'modelo_em_uso_continua_baseline':True,'modelo_substituido':False}

def compare_pair(log_c,log_m,expected_schedule):
    if not(len(log_c)==len(log_m)==len(expected_schedule)):raise ValueError('Quantidade de batches diferente entre braços.')
    home7_ignored=0;oi7_targets=0
    for a,b,p in zip(log_c,log_m,expected_schedule):
        for k in ['step','epoch','ids','input_signature']:
            if a[k]!=b[k] or a[k]!=p[k]:raise ValueError('Amostragem pareada divergente: '+k)
        if a['tensor_amostra_sha256']!=b['tensor_amostra_sha256']:raise ValueError('As amostras de pixels dos batches divergem.')
        if a['source_counts']!=b['source_counts']:raise ValueError('Exposições divergentes.')
        if a['gradient_audit']['unknown_negative_max']!=0 or b['gradient_audit']['unknown_negative_max']!=0:
            raise ValueError('Gradiente negativo vazou em classe desconhecida.')
        if b['gradient_audit']['home7_max']!=0:raise ValueError('Gradiente de wardrobe Home não foi zerado.')
        for i,iid in enumerate(a['ids']):
            ca={str(k):v for k,v in a['gt_per_image_by_class'][i].items()}
            cb={str(k):v for k,v in b['gt_per_image_by_class'][i].items()}
            if iid.startswith('home:'):
                home7_ignored+=ca.get('7',0)
                if cb.get('7',0) or a['known_masks'][i]!=4095 or b['known_masks'][i]!=3967:
                    raise ValueError('Supervisão Home7 diferente do plano.')
                ca.pop('7',None);cb.pop('7',None)
            else:
                oi7_targets+=ca.get('7',0)
                if a['known_masks'][i]!=b['known_masks'][i]:raise ValueError('Máscara Open Images alterada.')
            if ca!=cb:raise ValueError('Outros alvos diferem entre braços no batch observado.')
    if oi7_targets<=0:raise ValueError('Nenhum alvo wardrobe Open Images foi exposto ao treino.')
    return {'pareamento_conferido':True,'batches_por_braco':len(log_c),
            'mesmos_ids_na_mesma_ordem':True,'mesmos_hashes_imagem':True,
            'amostras_tensor_iguais':True,'gradiente_negativo_desconhecido_zero':True,
            'gradiente_home7_experimental_zero':True,
            'exposicoes_home7_ignoradas_somente_M':home7_ignored,
            'exposicoes_oi7_preservadas_em_ambos':oi7_targets,
            'outros_alvos_e_mascaras_preservados':True,
            'tensor_limite':'SHA de amostras espaciais de pixels, combinado a identidade das imagens e ausência de augmentações; não é um hash de cada pixel do tensor completo.'}
