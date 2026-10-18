"""Objetivos e orçamento da nova linha de desenvolvimento; sem teste final."""
from __future__ import annotations
import hashlib,json

SPEC={'version':'21-yolov8s-supervisao-ampliada-v1','model':'yolov8s','classes':12,'epochs':40,'patience':8,
    'imgsz':768,'batch':8,'home_per_batch':2,'oi_per_batch':6,'oi_pool_max':12000,'oi_pool_min':5000,
    'oi_candidates':15000,'per_primary_class':1250,'seed':21002026,'optimizer':'AdamW','lr0':.001,'lrf':.01,
    'warmup_epochs':3.,'weight_decay':.0005,'cos_lr':True,'nbs':32,'amp':False,
    'mosaic':0.,'mixup':0.,'cutmix':0.,'copy_paste':0.,'degrees':0.,'translate':.1,'scale':.3,
    'fliplr':.5,'flipud':0.,'hsv_h':.015,'hsv_s':.4,'hsv_v':.3,'workers':2,
    'oi_uniform_fraction':.75,'session_hours':8.,'target_precision':.80,'target_recall':.70,
    'thresholds':[.05,.10,.15,.20,.25,.30,.40,.50,.60,.70,.80,.90],
    'primary_cohort':'openimages_val_nova','inference_conf':.05,'nms_iou':.7,'match_iou':.5,'max_det':300,
    'fitness':'max_limiar(min(P/0.80,R/0.70)+0.000001*F1), detecção externa com cobertura; empate pelo menor limiar',
    'selection':'melhor fitness externo durante o treino; empate pela época anterior',
    'final_test_consulted':False,'historical_300_used_for_tuning':False}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def order_id(iid,seed):return hashlib.sha256(f'{seed}|{iid}'.encode()).hexdigest(),iid

def choose_candidates(history,forbidden_ids,forbidden_hashes,spec=SPEC):
    from collections import defaultdict
    eligible=[r for r in history if r['split']=='train' and r['id'] not in forbidden_ids and r['imagem_sha256'] not in forbidden_hashes]
    groups=defaultdict(list)
    for row in eligible:groups[int(row['classe_primaria'])].append(row)
    for cid in groups:groups[cid].sort(key=lambda r:order_id(r['id'],spec['seed']))
    chosen={}
    for k in range(spec['per_primary_class']):
        for cid in sorted(groups,key=lambda c:(len(groups[c]),c)):
            if len(chosen)>=spec['oi_candidates']:break
            if k<len(groups[cid]):chosen[groups[cid][k]['id']]=groups[cid][k]
    for row in sorted(eligible,key=lambda r:order_id(r['id'],spec['seed'])):
        if len(chosen)>=spec['oi_candidates']:break
        chosen.setdefault(row['id'],row)
    if len(chosen)<spec['oi_pool_min']:raise ValueError('Menos de 5 mil exemplos elegíveis no TRAIN histórico.')
    return sorted(chosen.values(),key=lambda r:r['id'])

def balanced_pool(records,spec=SPEC):
    from collections import defaultdict
    groups=defaultdict(list)
    for row in records:
        for cid in {g['classe_id'] for g in row['gt']}:groups[cid].append(row)
    for cid in groups:groups[cid].sort(key=lambda r:order_id(r['id'],spec['seed']+1))
    chosen={};limit=min(spec['oi_pool_max'],len(records));k=0
    while len(chosen)<limit:
        progress=False
        for cid in sorted(groups,key=lambda c:(len(groups[c]),c)):
            if k<len(groups[cid]):
                row=groups[cid][k];chosen.setdefault(row['id'],row);progress=True
                if len(chosen)>=limit:break
        if not progress:break
        k+=1
    if len(chosen)<spec['oi_pool_min']:raise ValueError('A preparação não encontrou supervisão suficiente para a nova rodada.')
    return sorted(chosen.values(),key=lambda r:r['id'])
