"""YOLO e CLIP são sinais de triagem em TRAIN, não validadores de verdade."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

import audit15_core as core

CLIP_COMMIT = 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
CONCEPTS = [
    ('bed', 'a bed with a mattress'),
    ('sofa', 'a sofa or couch'),
    ('chair', 'a chair, stool or armchair for one person'),
    ('table', 'a table, dining table, coffee table or desk'),
    ('lamp', 'a lamp or indoor light fixture'),
    ('tv', 'a television screen'),
    ('laptop', 'a laptop computer'),
    ('wardrobe', 'a wardrobe or tall closet for storing clothes'),
    ('window', 'a window in a building'),
    ('door', 'a door in a building'),
    ('potted plant', 'a real potted plant'),
    ('photo frame', 'a picture frame or framed photograph'),
    ('generic cabinet', 'a generic storage cabinet or cupboard'),
    ('bookcase', 'a bookcase or open shelving unit'),
    ('tv stand', 'a low TV stand or media console'),
    ('chest of drawers', 'a chest of drawers or dresser'),
    ('sideboard', 'a sideboard or buffet cabinet'),
    ('refrigerator', 'a refrigerator'),
    ('radiator', 'a radiator or space heater'),
    ('outdoor vegetation', 'outdoor trees, bushes or grass'),
    ('person', 'a person'),
    ('wall or floor', 'an empty wall or floor'),
    ('decorative object', 'a vase, ornament or decorative object'),
    ('unrecognizable detail', 'an unrecognizable close-up detail or texture'),
]
TEMPLATES = ['a photo of {}.', 'a cropped photo of {}.', 'an image showing {}.']
SCENES = [
    ('interior', 'a photo of the inside of a furnished room'),
    ('exterior', 'a photo of an outdoor scene, street or garden'),
    ('produto_fundo_liso', 'a studio product photograph on a plain background'),
    ('colagem_ou_desenho', 'a collage, illustration or drawing'),
    ('detalhe_sem_contexto', 'a close-up photograph with no visible room context'),
]
PREDICT_ARGS = {'imgsz':640,'conf':.15,'iou':.70,'max_det':300,
                'half':False,'augment':False,'rect':True,'verbose':False}
SEMANTIC_RULE = {'views':[1.0,1.2],'min_crop_side':8,'min_cosine':.20,
                 'min_margin_over_label_both_views':.03,
                 'unmatched_focus_min_score':.25,'unmatched_per_image_limit':10,
                 'scene_min_cosine':.20,'scene_margin':.02,
                 'thresholds_calibrated':False}


def environment():
    import torch, torchvision, ultralytics, PIL, scipy
    return {'torch':torch.__version__,'torchvision':torchvision.__version__,
            'ultralytics':ultralytics.__version__,'numpy':np.__version__,
            'pillow':PIL.__version__,'scipy':scipy.__version__,
            'ftfy':importlib.metadata.version('ftfy'),'regex':importlib.metadata.version('regex'),
            'clip_commit':CLIP_COMMIT,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def validate_prediction_row(row):
    for p in row['predictions']:
        c, score, box = p['classe_id'], p['confianca'], p['bbox']
        if (type(c) is not int or not 0 <= c < 12 or not isinstance(score,(int,float))
                or not math.isfinite(score) or not 0 <= score <= 1 or len(box)!=4
                or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in box)
                or box[2]<box[0] or box[3]<box[1]):
            raise ValueError('Previsão estruturalmente inválida: '+row['id'])


def positive_area(p):
    b=p['bbox']; return b[2]>b[0] and b[3]>b[1]


def train_guard(records):
    if any(r.get('role')!='train' or r.get('status_input')!='legivel' or not r.get('image') for r in records):
        raise ValueError('Somente treino legível e sem quarentena pode entrar nos modelos.')


def matches(gt,pred,threshold=.5):
    matrix=core.iou_matrix(gt,pred)
    options=[]
    for gi,g in enumerate(gt):
        for pi,p in enumerate(pred):
            if g['classe_id']==p['classe_id'] and matrix[gi,pi]>=threshold:
                options.append((-float(matrix[gi,pi]),gi,pi))
    pairs=[]; used_g=set(); used_p=set()
    for neg,gi,pi in sorted(options):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi);used_p.add(pi);pairs.append((gi,pi,-neg))
    return pairs, [i for i in range(len(gt)) if i not in used_g], [i for i in range(len(pred)) if i not in used_p]


def detector_evidence(record,predictions):
    # Todas as classes anotadas são associadas; cobertura restringe a interpretação das sobras.
    pred=[p for p in predictions if p['confianca']>=.25 and positive_area(p)]
    pairs,missing,extras=matches(record['gt'],pred)
    gmatch={gi:(pi,overlap) for gi,pi,overlap in pairs}
    gt_rows=[]
    for gi,g in enumerate(record['gt']):
        row={'gt_index':gi,'matched_at_025':gi in gmatch,'matched_score':None,'iou':None}
        if gi in gmatch:
            pi,overlap=gmatch[gi];row.update(matched_score=pred[pi]['confianca'],iou=overlap)
        gt_rows.append(row)
    extra_rows=[]
    for pi in extras:
        p=pred[pi]; known=bool(record['known_mask'] & (1 << p['classe_id']))
        overlap=core.iou_matrix(record['gt'],[p])[:,0]
        best=int(np.argmax(overlap)) if len(overlap) else None
        extra_rows.append({**p,'known_class':known,'best_iou_any_label':float(overlap[best]) if best is not None else 0.,
                           'overlapping_label_class':record['gt'][best]['classe_id'] if best is not None else None,
                           'interpretation':'sobra_em_classe_coberta_a_investigar' if known else 'cobertura_desconhecida_nao_contar_como_erro'})
    selected=sorted([p for p in extra_rows if p['classe_id'] in core.FOCUS],
                    key=lambda p:(-p['confianca'],p['classe_id'],p['bbox']))[:SEMANTIC_RULE['unmatched_per_image_limit']]
    return {'gt':gt_rows,'extras':extra_rows,'extra_crops':selected,
            'gt_without_match_at_025':len(missing),
            'zero_area_proposals':sum(not positive_area(p) for p in predictions),
            'saturated_max_det':len(predictions)>=PREDICT_ARGS['max_det']}


def infer_yolo(records,weight,expected_sha,cache_dir,context,device,checkpoint=None,predict_fn=None):
    train_guard(records)
    if core.sha_file(weight)!=expected_sha: raise ValueError('Peso B alterado antes da inferência.')
    model=None
    if predict_fn is None:
        from ultralytics import YOLO
        model=YOLO(str(weight))
        names=model.names
        if isinstance(names,list): names=dict(enumerate(names))
        if {int(k):v for k,v in names.items()}!=dict(enumerate(core.NAMES)): raise ValueError('Classes dos pesos incompatíveis.')
        def predict_fn(r):
            result=model.predict(source=r['image'],device=device,**PREDICT_ARGS)[0]
            return [{'classe_id':int(c),'confianca':float(s),'bbox':[float(x) for x in b]}
                    for b,c,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist())]
    results={};done=0;fresh=0;t0=time.monotonic()
    for name,group in core.chunks(records):
        sig=core.digest({'context':context,'weight':expected_sha,'predict':PREDICT_ARGS,
                         'records':[core.record_identity(r) for r in group]})
        path=Path(cache_dir)/'yolo'/(name+'.json.gz')
        rows=core.recoverable_cache_read(path,sig,[r['id'] for r in group],validate_prediction_row)
        if rows is None:
            rows=[]
            for r in group:
                row={'id':r['id'],'predictions':predict_fn(r)};validate_prediction_row(row); rows.append(row);fresh+=1
            core.cache_write(path,sig,rows)
        results.update({r['id']:r['predictions'] for r in rows});done+=len(group)
        elapsed=time.monotonic()-t0;rate=fresh/max(elapsed,1e-9)
        eta=(len(records)-done)/rate/60 if rate else None
        message=f'YOLO: {done}/{len(records)}; novas inferências nesta passagem: {fresh}'
        if eta is not None:message+=f'; estimativa restante pela velocidade atual: {eta:.1f} min'
        print(message,flush=True)
        if checkpoint: checkpoint('cache YOLO '+name)
    del model
    return results,{'images':len(records),'new_inferences':fresh,'seconds':time.monotonic()-t0}


def crop_square(image,box,factor=1.):
    w,h=image.size;x1,y1,x2,y2=box
    cx=(x1+x2)/2;cy=(y1+y2)/2;bw=(x2-x1)*factor;bh=(y2-y1)*factor
    xy=[max(0,math.floor(cx-bw/2)),max(0,math.floor(cy-bh/2)),
        min(w,math.ceil(cx+bw/2)),min(h,math.ceil(cy+bh/2))]
    if xy[2]<=xy[0] or xy[3]<=xy[1]: return None
    region=image.crop(tuple(xy)).convert('RGB'); side=max(region.size)
    square=Image.new('RGB',(side,side),(127,127,127));square.paste(region,((side-region.width)//2,(side-region.height)//2))
    return square.resize((224,224),Image.Resampling.BICUBIC)


def semantic_result(scores,expected):
    a=np.asarray(scores,dtype=float)
    if a.shape!=(2,len(CONCEPTS)) or not np.isfinite(a).all() or np.any(np.abs(a)>1.0001):
        raise ValueError('Scores CLIP incompatíveis.')
    best=np.argmax(a,axis=1);top=[float(a[i,best[i]]) for i in range(2)]
    delta=[float(a[i,best[i]]-a[i,expected]) for i in range(2)]
    stable=bool(best[0]==best[1]); agree=bool(np.all(best==expected))
    strong=bool(stable and not agree and min(top)>=SEMANTIC_RULE['min_cosine']
                and min(delta)>=SEMANTIC_RULE['min_margin_over_label_both_views'])
    status='divergencia_forte_ia' if strong else ('compativel_ia' if agree else 'inconclusivo_ia')
    return {'status':status,'expected':core.NAMES[expected],
            'top_tight':CONCEPTS[int(best[0])][0],'top_context':CONCEPTS[int(best[1])][0],
            'top_cosine_tight':top[0],'top_cosine_context':top[1],
            'margin_over_label_tight':delta[0],'margin_over_label_context':delta[1],
            'scores':a.tolist(),'probability_calibrated':False}


def scene_result(scores):
    a=np.asarray(scores,dtype=float)
    if a.shape!=(len(SCENES),) or not np.isfinite(a).all() or np.any(np.abs(a)>1.0001):
        raise ValueError('Scores de domínio inválidos.')
    idx=np.argsort(-a,kind='stable');top=int(idx[0]);margin=float(a[top]-a[idx[1]])
    indicative=bool(a[top]>=SEMANTIC_RULE['scene_min_cosine'] and margin>=SEMANTIC_RULE['scene_margin'])
    return {'top':SCENES[top][0],'cosine':float(a[top]),'margin':margin,
            'indicative':indicative,'scores':a.tolist(),'probability_calibrated':False}


class ClipScorer:
    def __init__(self,device,download_root):
        import clip, torch
        import clip.clip as impl
        expected_url=impl._MODELS['ViT-B/32']
        if expected_url.split('/')[-2]!=CLIP_SHA: raise ValueError('Peso CLIP não corresponde ao hash congelado.')
        self.device='cuda:0' if device==0 else 'cpu';self.torch=torch;self.clip=clip
        self.model,self.preprocess=clip.load('ViT-B/32',device=self.device,jit=False,download_root=str(download_root))
        path=Path(download_root)/'ViT-B-32.pt'
        if core.sha_file(path)!=CLIP_SHA: raise ValueError('Hash do CLIP inválido.')
        self.model=self.model.float().eval()
        self.model.requires_grad_(False)
        self.batch_size=32 if device==0 else 8
        self.concepts=self._text_features([p for _,p in CONCEPTS])
        self.scenes=self._text_features([p for _,p in SCENES],templates=['{}'])

    def _text_features(self,descriptions,templates=TEMPLATES):
        torch=self.torch
        sentences=[t.format(d) for d in descriptions for t in templates]
        pieces=[]
        with torch.inference_mode():
            for i in range(0,len(sentences),32):
                x=self.model.encode_text(self.clip.tokenize(sentences[i:i+32]).to(self.device)).float()
                pieces.append(x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12))
            x=torch.cat(pieces).reshape(len(descriptions),len(templates),-1).mean(dim=1)
            return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)

    def score(self,images,scenes=False):
        torch=self.torch;features=[]
        if not images:return np.empty((0,len(SCENES) if scenes else len(CONCEPTS)))
        with torch.inference_mode():
            for i in range(0,len(images),self.batch_size):
                inp=torch.stack([self.preprocess(im) for im in images[i:i+self.batch_size]]).to(self.device)
                x=self.model.encode_image(inp).float();x=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                features.append((x @ (self.scenes if scenes else self.concepts).T).cpu().numpy())
        out=np.concatenate(features)
        if not np.isfinite(out).all():raise ValueError('CLIP retornou valores não finitos.')
        return out


def score_record(record,predictions,scorer):
    evidence=detector_evidence(record,predictions)
    jobs=[('gt',g['index'],g) for g in record['gt']]
    jobs += [('extra',i,p) for i,p in enumerate(evidence['extra_crops'])]
    rendered=[];positions=[];details=[]
    with Image.open(record['image']) as im:
        image=im.convert('RGB')
        # A cena inteira é reduzida com padding, preservando os cantos da imagem.
        scene_img=crop_square(image,[0,0,*image.size])
        scene=scene_result(scorer.score([scene_img],scenes=True)[0])
        for kind,index,obj in jobs:
            box=obj['bbox'];entry={'kind':kind,'index':index,'classe_id':obj['classe_id'],'bbox':box}
            if min(box[2]-box[0],box[3]-box[1])<SEMANTIC_RULE['min_crop_side']:
                details.append({**entry,'status':'recorte_pequeno_sem_analise','scores':None});continue
            crops=[crop_square(image,box,factor) for factor in SEMANTIC_RULE['views']]
            if any(c is None for c in crops):
                details.append({**entry,'status':'recorte_invalido_sem_analise','scores':None});continue
            positions.append((entry,len(rendered)));rendered.extend(crops)
        # Inferência em minibatches dentro de score; uma imagem é aberta uma só vez.
        scores=scorer.score(rendered)
        for entry,index in positions:details.append({**entry,**semantic_result(scores[index:index+2],entry['classe_id'])})
    return {'id':record['id'],'scene':scene,'objects':sorted(details,key=lambda x:(x['kind'],x['index'])),
            'detector':evidence}


def validate_semantic_row(row):
    scene_result(row['scene']['scores'])
    for obj in row['objects']:
        if obj.get('scores') is not None:
            recalculated=semantic_result(obj['scores'],obj['classe_id'])
            if recalculated['status']!=obj['status']:raise ValueError('Decisão CLIP incompatível com scores no cache.')


def infer_semantics(records,predictions,cache_dir,context,device,model_dir,checkpoint=None,scorer=None):
    train_guard(records);results={};fresh=0;done=0;start=time.monotonic()
    for name,group in core.chunks(records):
        sig=core.digest({'context':context,'clip_sha':CLIP_SHA,'concepts':CONCEPTS,'templates':TEMPLATES,
                         'scenes':SCENES,'rule':SEMANTIC_RULE,'records':[core.record_identity(r) for r in group],
                         'predictions_sha256':core.digest({r['id']:predictions[r['id']] for r in group})})
        path=Path(cache_dir)/'clip'/(name+'.json.gz')
        rows=core.recoverable_cache_read(path,sig,[r['id'] for r in group],validate_semantic_row)
        if rows is None:
            if scorer is None:scorer=ClipScorer(device,model_dir)
            rows=[]
            for r in group:
                row=score_record(r,predictions[r['id']],scorer);validate_semantic_row(row);rows.append(row);fresh+=1
                if len(rows)%10==0:print(f'CLIP: bloco {name}, {len(rows)}/{len(group)} imagens',flush=True)
            core.cache_write(path,sig,rows)
        results.update({r['id']:r for r in rows});done+=len(group)
        speed=fresh/max(time.monotonic()-start,1e-9)
        msg=f'CLIP: {done}/{len(records)} imagens; novas análises nesta passagem: {fresh}'
        if speed:msg+=f'; estimativa restante pela velocidade atual: {(len(records)-done)/speed/60:.1f} min'
        print(msg,flush=True)
        if checkpoint:checkpoint('cache CLIP '+name)
    return results,{'images':len(records),'new_analyses':fresh,'seconds':time.monotonic()-start}
