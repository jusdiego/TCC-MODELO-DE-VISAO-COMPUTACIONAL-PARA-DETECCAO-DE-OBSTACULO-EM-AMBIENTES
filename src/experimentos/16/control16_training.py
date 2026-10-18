"""Treino pareado com auditoria de batches e gradiente da BCE parcial."""
from __future__ import annotations
import hashlib,json,random,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import unwrap_model
import diag11_core as core
from exp10_partial import BCEComCobertura,DetectionLossParcial
from exp12_training import ControlledTrainer,state_fingerprint
import control16_protocol as p16


class AuditedBCE(BCEComCobertura):
    def __init__(self,original=None):
        super().__init__(original);self.home_mask=None;self.audit_step=0;self.last_audit=None

    def forward(self,logits,target):
        result=super().forward(logits,target)
        if logits.requires_grad:
            known=self.known.detach().clone();home=self.home_mask.detach().clone();step=self.audit_step
            unknown_negative=(~known[:,None,:]) & (target.detach()==0)
            positives_unknown=(~known[:,None,:]) & (target.detach()>0)
            if (home[:,None] & ~known[:,7:8]).any() and target.detach()[home,:,7].ne(0).any():
                raise RuntimeError('Alvo positivo wardrobe Home permaneceu na perda mascarada.')
            def check_gradient(grad):
                max_unknown=float(grad[unknown_negative].abs().max().item()) if unknown_negative.any() else 0.
                home7=float(grad[home,:,7].abs().max().item()) if home.any() else 0.
                if max_unknown!=0:raise RuntimeError('Gradiente negativo não zero em classe desconhecida.')
                if home.any() and (~known[home,7]).all() and home7!=0:
                    raise RuntimeError('Gradiente classificador de wardrobe Home não foi zerado.')
                self.last_audit={'batch_step':step,'unknown_negative_max':max_unknown,'home7_max':home7,
                                 'unknown_negative_elements':int(unknown_negative.sum().item()),
                                 'unknown_positive_elements':int(positives_unknown.sum().item())}
                return grad
            logits.register_hook(check_gradient)
        return result


class SourceMaskTrainer(ControlledTrainer):
    arm=None;expected_schedule=None;attempt_dir=None;expected_initial_state=None

    def __init__(self,*args,**kwargs):
        self.batch_log=[];self.pending_batch=None
        super().__init__(*args,**kwargs)
        if self.arm not in p16.ARMS or self.expected_schedule is None:raise RuntimeError('Braço sem protocolo pareado.')
        if len(self.expected_schedule)!=self.control_spec['epochs']*self.control_spec['batches']:
            raise RuntimeError('Cronograma incompleto.')
        self.add_callback('on_train_start',verify_start)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode=='train':
            if any(float(getattr(self.args,k,0))!=0 for k in p16.AUGMENTATION_ZERO):
                raise RuntimeError('Todas as augmentações devem ser zero neste pareamento.')
            if self.args.workers!=0 or self.args.multi_scale or self.args.rect:
                raise RuntimeError('Use workers=0, sem multi_scale e sem rect no treino pareado.')
        loader=super().get_dataloader(dataset_path,batch_size,rank,mode)
        if mode=='train':
            actual=[self.source_records[Path(f).resolve().as_posix()]['id'] for f in loader.dataset.im_files]
            expected=[r['id'] for r in p16.logical_order(list(self.source_records.values()))]
            if actual!=expected:raise RuntimeError('Ordem de imagens do loader diverge do plano congelado.')
            # A biblioteca elimina linhas exatamente repetidas em seu cache; confira o conjunto de alvos.
            for label in loader.dataset.labels:
                record=self.source_records[Path(label['im_file']).resolve().as_posix()]
                raw=Path(record['label']).read_text(encoding='utf-8-sig')
                rows=np.array([list(map(float,l.split())) for l in raw.splitlines() if l.strip()],dtype=np.float32).reshape(-1,5)
                rows=np.unique(rows,axis=0)
                actual_rows=np.concatenate([label['cls'],label['bboxes']],axis=1).astype(np.float32)
                def canon(x):return sorted(tuple(float(v) for v in row) for row in x)
                if canon(rows)!=canon(actual_rows):raise RuntimeError('O loader alterou os alvos projetados.')
        return loader

    def preprocess_batch(self,batch):
        result=super().preprocess_batch(batch)
        step=self.batches_seen;planned=self.expected_schedule[step-1]
        rs=[self.source_records[Path(f).resolve().as_posix()] for f in result['im_file']]
        ids=[r['id'] for r in rs];signature=p16.digest([(r['id'],r['sha256']) for r in rs])
        if ids!=planned['ids'] or signature!=planned['input_signature'] or self.epoch+1!=planned['epoch']:
            raise RuntimeError('Batch observado difere do cronograma pareado.')
        home=torch.tensor([r['source']=='home' for r in rs],device=self.device,dtype=torch.bool)
        masks=[r['known_mask'] for r in rs]
        cls=result['cls'].flatten().long();bi=result['batch_idx'].long()
        if self.arm==p16.ARMS[1]:
            if result['class_known'][home,7].any() or any(int(cls[j])==7 and bool(home[int(bi[j])]) for j in range(len(cls))):
                raise RuntimeError('Wardrobe Home foi mantida como alvo ou negativo conhecido.')
        elif not result['class_known'][home,7].all():raise RuntimeError('O controle perdeu cobertura Home.')
        criterion=unwrap_model(self.model).criterion
        if not isinstance(criterion,DetectionLossParcial):raise RuntimeError('Perda parcial não instalada.')
        if not isinstance(criterion.bce,AuditedBCE):criterion.bce=AuditedBCE(criterion.bce.original)
        criterion.bce.home_mask=home;criterion.bce.audit_step=step;criterion.bce.last_audit=None
        sample=result['img'][:,:,::32,::32].detach().cpu().float().contiguous().numpy().tobytes()
        gt_counts=[]
        for i in range(len(rs)):
            gt_counts.append(dict(Counter(int(v) for v in cls[bi==i].tolist())))
        self.pending_batch={**planned,'source_counts':dict(Counter(r['source'] for r in rs)),
            'known_masks':masks,'gt_per_image_by_class':gt_counts,
            'tensor_amostra_sha256':hashlib.sha256(sample).hexdigest()}
        return result

    def optimizer_step(self):
        audit=unwrap_model(self.model).criterion.bce.last_audit
        if audit is None or audit['batch_step']!=self.batches_seen:raise RuntimeError('Backward não foi auditado para o batch atual.')
        if self.arm==p16.ARMS[1] and audit['home7_max']!=0:raise RuntimeError('Máscara wardrobe Home deixou passar gradiente.')
        super().optimizer_step()
        row={**self.pending_batch,'gradient_audit':audit}
        if row['step']!=self.actual_updates:raise RuntimeError('Mais de uma atualização ou batch saltado.')
        self.batch_log.append(row)
        path=Path(self.attempt_dir)/'auditoria_batches.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as f:f.write(p16.canonical(row)+'\n')

    def validate(self):
        # As duas coortes são avaliadas pelo callback externo fixo, no checkpoint EMA exportado.
        return self.metrics,0.

    def final_eval(self):
        return None


def verify_start(trainer):
    actual=state_fingerprint(trainer.model)
    if actual!=trainer.expected_initial_state:raise RuntimeError('Estado inicial diferente do candidato 10.')
    core.save_json(Path(trainer.attempt_dir)/'inicializacao.json',{
        'arm':trainer.arm,'state_sha256':actual,'same_initial_weights':True})


def self_test(device='cpu'):
    """Backward real da BCE e da perda YOLO, usando somente tensores sintéticos."""
    py=random.getstate();ns=np.random.get_state();ts=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        logits=torch.zeros((2,4,12),device=device,requires_grad=True);targets=torch.zeros_like(logits)
        targets[1,0,7]=1;known=torch.ones((2,12),dtype=torch.bool,device=device);known[0,7]=False;known[1,10]=False
        bce=AuditedBCE();bce.known=known;bce.home_mask=torch.tensor([True,False],device=device);bce.audit_step=1
        bce(logits,targets).sum().backward()
        assert logits.grad[0,:,7].eq(0).all() and logits.grad[1,0,7]<0 and logits.grad[1,:,10].eq(0).all()
        ordinary=nn.BCEWithLogitsLoss(reduction='none')
        full=BCEComCobertura();full.known=torch.ones_like(known)
        torch.testing.assert_close(full(logits.detach(),targets),ordinary(logits.detach(),targets))
        # Integração com assignment e termos de caixa/DFL da versão exata do YOLO.
        model=DetectionModel('yolov8n.yaml',nc=12,verbose=False).to(device).train()
        from ultralytics.cfg import get_cfg
        model.args=get_cfg();criterion=DetectionLossParcial(model)
        criterion.bce=AuditedBCE();criterion.bce.home_mask=torch.tensor([True,False],device=device);criterion.bce.audit_step=2
        batch={'img':torch.rand(2,3,64,64,device=device),
               'batch_idx':torch.tensor([0,1],device=device),
               'cls':torch.tensor([[2.],[7.]],device=device),
               'bboxes':torch.tensor([[.5,.5,.5,.5],[.5,.5,.4,.4]],device=device),
               'class_known':known}
        loss,items=criterion(model(batch['img']),batch);loss.sum().backward()
        assert torch.isfinite(loss).all() and criterion.bce.last_audit['home7_max']==0
        assert criterion.bce.last_audit['unknown_negative_max']==0
        result={'bce_full_matches_standard':True,'home7_gradient_zero':True,'openimages7_positive_preserved':True,
                'unknown_negative_gradient_zero':True,'real_yolo_loss_backward_finite':True,
                'device':str(device),'synthetic_only':True}
        del model,criterion,batch,loss
        return result
    finally:
        random.setstate(py);np.random.set_state(ns);torch.set_rng_state(ts)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
