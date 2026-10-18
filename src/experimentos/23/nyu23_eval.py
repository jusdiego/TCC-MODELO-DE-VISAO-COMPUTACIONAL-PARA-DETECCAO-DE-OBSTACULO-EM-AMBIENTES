"""Seleção em VAL e teste congelado; métricas micro por objeto com IoU 0,5."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import gzip,json,math
import numpy as np
import nyu22_common as c

def metrics(tp,fn,fp):
    tp,fn,fp=map(int,(tp,fn,fp));p=tp/(tp+fp) if tp+fp else None;r=tp/(tp+fn) if tp+fn else None
    return {'tp':tp,'fn':fn,'fp':fp,'precision':p,'recall':r,'f1':2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else None}
def match(gt,pred,threshold=.5):
    if not gt or not pred:return [],list(range(len(gt))),list(range(len(pred)))
    a=np.asarray([o['bbox'] for o in gt],float);b=np.asarray([o['bbox'] for o in pred],float)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.prod(np.maximum(hi-lo,0),axis=2);aa=np.prod(np.maximum(a[:,2:]-a[:,:2],0),axis=1);bb=np.prod(np.maximum(b[:,2:]-b[:,:2],0),axis=1)
    den=aa[:,None]+bb[None,:]-inter;overlap=np.divide(inter,den,out=np.zeros_like(inter),where=den>0)
    candidates=sorted([(-float(overlap[i,j]),i,j) for i,g in enumerate(gt) for j,p in enumerate(pred) if g['class_id']==p['class_id'] and overlap[i,j]>=threshold])
    gs=set();ps=set();pairs=[]
    for neg,i,j in candidates:
        if i not in gs and j not in ps:gs.add(i);ps.add(j);pairs.append((i,j,-neg))
    return pairs,[i for i in range(len(gt)) if i not in gs],[j for j in range(len(pred)) if j not in ps]

def profile(rows,cache,threshold,spec=c.TRAIN_SPEC):
    totals=np.zeros(3,dtype=np.int64);classes=np.zeros((6,3),dtype=np.int64);details=[]
    for row in rows:
        pred=[p for p in cache[row['id']] if p['confidence']>=threshold];gt=row['gt']
        pairs,missing,extras=match(gt,pred,spec['match_iou']);counts=[len(pairs),len(missing),len(extras)];totals+=counts
        for gi,_,_ in pairs:classes[gt[gi]['class_id'],0]+=1
        for gi in missing:classes[gt[gi]['class_id'],1]+=1
        for pi in extras:classes[pred[pi]['class_id'],2]+=1
        details.append({'id':row['id'],'scene_group':row['scene_group'],'tp':counts[0],'fn':counts[1],'fp':counts[2]})
    det=metrics(*totals)
    score=min((det['precision'] or 0)/spec['target_precision'],(det['recall'] or 0)/spec['target_recall'])+1e-6*(det['f1'] or 0)
    return {'images':len(rows),'threshold':threshold,'detector':det,'fitness':score,
        'target_met':bool(det['precision'] is not None and det['recall'] is not None and det['precision']>=spec['target_precision'] and det['recall']>=spec['target_recall']),
        'classes':[{'class_id':i,'class':name,'gt':int(classes[i,0]+classes[i,1]),**metrics(*classes[i])} for i,name in enumerate(c.NAMES)],'image_rows':details}
def choose(profiles):return max(profiles,key=lambda v:(v['fitness'],-v['threshold']))

def predict(model,rows,device,spec):
    if list(model.names.values())!=c.NAMES:raise ValueError('O peso não contém as seis classes na ordem esperada.')
    batch=spec['predict_batch'];cache={};diagnostic={'zero_area':0,'outside':0,'saturated_images':0,'coordinates_changed':False,'predictions_removed':0}
    for start in range(0,len(rows),batch):
        block=rows[start:start+batch]
        result=model.predict([str(r['image']) for r in block],conf=spec['inference_conf'],imgsz=spec['imgsz'],rect=True,
            iou=spec['nms_iou'],max_det=spec['max_det'],quantize=32,batch=batch,device=device,verbose=False,save=False)
        if len(result)!=len(block):raise ValueError('Número de resultados diferente do bloco de imagens.')
        for r,out in zip(block,result):
            if Path(out.path).resolve()!=Path(r['image']).resolve() or tuple(out.orig_shape)!=(r['height'],r['width']):raise ValueError('Previsão pertence a outra imagem.')
            values=[]
            if out.boxes is not None:
                for cls,xy,confidence in zip(out.boxes.cls.cpu().numpy(),out.boxes.xyxy.cpu().numpy(),out.boxes.conf.cpu().numpy()):
                    cid=int(cls);xy=[float(x) for x in xy];confidence=float(confidence)
                    if not(0<=cid<6 and all(math.isfinite(x) for x in xy+[confidence]) and 0<=confidence<=1 and xy[2]>=xy[0] and xy[3]>=xy[1]):raise ValueError('Previsão estruturalmente inválida.')
                    diagnostic['zero_area']+=int(xy[2]==xy[0] or xy[3]==xy[1]);diagnostic['outside']+=int(min(xy)<-.01 or xy[2]>r['width']+.01 or xy[3]>r['height']+.01)
                    values.append({'class_id':cid,'bbox':xy,'confidence':confidence})
            cache[r['id']]=values;diagnostic['saturated_images']+=int(len(values)==spec['max_det'])
    return cache,diagnostic

def obtain_cache(model,rows,device,spec,output,weight_sha,role,checkpoint=None):
    if any(r['role']!=role for r in rows):raise ValueError('Papel proibido nesta avaliação.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    request={'role':role,'weight_sha256':weight_sha,'input_sha256':c.digest([{k:r[k] for k in ['id','image_sha256','label_sha256','width','height']} for r in rows]),
        'inference':{k:spec[k] for k in ['imgsz','inference_conf','nms_iou','max_det','predict_batch']},'quantize':32}
    key=c.digest(request);output=output/key[:24];output.mkdir(parents=True,exist_ok=True);cache={};diagnostics=[]
    for start in range(0,len(rows),32):
        block=rows[start:start+32];path=output/f'bloco_{start//32:04d}.json.gz'
        if path.is_file():
            with gzip.open(path,'rt') as f:payload=json.load(f)
            if payload['request_key']!=key or payload['data_sha256']!=c.digest(payload['data']) or list(payload['data'])!=[r['id'] for r in block]:raise ValueError('Cache de previsão alterado ou de outra execução.')
        else:
            data,diagnostic=predict(model,block,device,spec);payload={'request_key':key,'data':data,'data_sha256':c.digest(data),'diagnostic':diagnostic}
            tmp=path.with_suffix('.tmp')
            with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,allow_nan=False)
            tmp.replace(path)
        cache.update(payload['data']);diagnostics.append(payload['diagnostic'])
        print(f'Avaliação {role}: {start+len(block)}/{len(rows)} imagens',flush=True)
        if checkpoint:checkpoint()
    c.save(output/'manifesto_cache.json',{'request':request,'request_key':key,'complete':True,'diagnostics':diagnostics})
    return cache

def evaluate_development(model,rows,device,spec=c.TRAIN_SPEC,output=None,weight_sha=None,fixed_threshold=None):
    if any(r['role']!='val' for r in rows):raise ValueError('A seleção admite somente VAL.')
    if output is None:cache,_=predict(model,rows,device,spec)
    else:cache=obtain_cache(model,rows,device,spec,Path(output)/'cache',weight_sha,'val')
    profiles=[profile(rows,cache,t,spec) for t in spec['thresholds']]
    selected=choose(profiles) if fixed_threshold is None else profile(rows,cache,fixed_threshold,spec)
    if output is not None:
        c.save(Path(output)/'metricas.json',selected);c.save(Path(output)/'grade_limiares.json',profiles)
        c.csv_save(Path(output)/'classes.csv',selected['classes']);c.csv_save(Path(output)/'imagens.csv',selected['image_rows'])
    return selected,profiles

def scene_bootstrap(image_rows,spec=c.TRAIN_SPEC):
    groups=defaultdict(lambda:np.zeros(3,dtype=np.int64))
    for r in image_rows:groups[r['scene_group']]+=np.asarray([r['tp'],r['fn'],r['fp']])
    values=np.array(list(groups.values()));rng=np.random.default_rng(spec['seed']);samples={k:[] for k in ['precision','recall','f1']}
    for _ in range(spec['bootstrap_groups']):
        result=metrics(*values[rng.integers(0,len(values),size=len(values))].sum(axis=0))
        for k in samples:
            if result[k] is not None:samples[k].append(result[k])
    return {'method':'Bootstrap de cenas, não de caixas individuais','groups':len(values),'replicates':spec['bootstrap_groups'],
        'interval_95':{k:[float(x) for x in np.quantile(v,[.025,.975])] if v else None for k,v in samples.items()}}
