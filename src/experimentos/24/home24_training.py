"""RF-DETR 1.11.2: AMP, EMA, orçamento medido e dois checkpoints completos."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import gc,json,math,shutil,time
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Checkpoint,ModelCheckpoint
from rfdetr.config import RFDETRSmallConfig,TrainConfig
from rfdetr.training import RFDETRModelModule,RFDETRDataModule,build_trainer
from rfdetr.training.auto_batch import resolve_auto_batch_config
import home24_common as c

class EpochPersistence(Checkpoint):
    """Executa depois da validação; publica metadados somente após os bytes."""
    def __init__(self,output,store,remote,context,mc,tc,prior=None,budget_seconds=8400):
        super().__init__();self.output=Path(output);self.store=store;self.remote=remote;self.context=context
        self.mc=mc;self.tc=tc;self.prior=prior or {};self.elapsed=float(self.prior.get('elapsed_seconds',0))
        self.budget=float(self.prior.get('budget_seconds',budget_seconds));self.start=None;self.epoch_start=None
        self.history=c.read(self.output/'historico_24.json') if (self.output/'historico_24.json').is_file() else []
        self.best=c.read(self.output/'melhor_24.json') if (self.output/'melhor_24.json').is_file() else None
        self.reason=None;self.base_elapsed=self.elapsed;self.latest=None
    @property
    def state_key(self):return 'home24-persistencia-'+self.context[:24]
    def state_dict(self):return {'elapsed_seconds':self.total_elapsed(),'budget_seconds':self.budget,'context':self.context}
    def load_state_dict(self,state):
        if state.get('context')!=self.context:raise ValueError('Checkpoint de outro protocolo.')
        self.elapsed=max(self.elapsed,float(state['elapsed_seconds']));self.base_elapsed=self.elapsed
        if not math.isclose(float(state['budget_seconds']),self.budget):raise ValueError('Orçamento do checkpoint mudou.')
    def total_elapsed(self):return self.base_elapsed+(time.monotonic()-self.start if self.start is not None else 0)
    def on_fit_start(self,trainer,module):self.base_elapsed=self.elapsed;self.start=time.monotonic()
    def on_train_epoch_start(self,trainer,module):self.epoch_start=time.monotonic()
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        value=outputs.get('loss') if isinstance(outputs,dict) else outputs
        if isinstance(value,torch.Tensor) and not torch.isfinite(value.detach()).all():
            raise FloatingPointError('Loss não finita; último checkpoint completo preservado.')
    def export_ema(self,trainer,module,epoch,score):
        state=None
        for callback in trainer.callbacks:
            getter=getattr(callback,'get_ema_model_state_dict',None)
            if callable(getter):state=getter();break
        if state is None:raise RuntimeError('EMA indisponível: não será trocada silenciosamente pelo modelo regular.')
        if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
            raise FloatingPointError('EMA não finita; melhor peso anterior preservado.')
        state={k:v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu() for k,v in state.items()}
        args={**self.mc.model_dump(),**self.tc.model_dump(),'class_names':c.NAMES}
        payload={'model':state,'args':args,'epoch':epoch-1,'model_config':self.mc.model_dump(),
          'model_name':'RFDETRSmall','class_names':c.NAMES,'context':self.context,'val_map50_95':score}
        p=self.output/'treino'/'melhor_ema.pth';t=p.with_suffix('.tmp');torch.save(payload,t);t.replace(p)
        info={'epoch':epoch,'metric':'val/ema_mAP_50_95','value':score,'weight_sha256':c.sha(p),
          'context':self.context,'weight':'treino/melhor_ema.pth','class_names':c.NAMES,'optimizer_included':False}
        c.save(self.output/'melhor_24.json',info);self.best=info
    def on_train_epoch_end(self,trainer,module):
        epoch=trainer.current_epoch+1
        values={k:float(v.detach().cpu()) if isinstance(v,torch.Tensor) else float(v) for k,v in trainer.callback_metrics.items() if isinstance(v,(int,float,torch.Tensor)) and (not isinstance(v,torch.Tensor) or v.numel()==1)}
        score=values.get('val/ema_mAP_50_95',values.get('val/mAP_50_95'))
        if score is None or not math.isfinite(score):raise ValueError('Época sem mAP de validação finito.')
        if self.best is None or score>self.best['value']:self.export_ema(trainer,module,epoch,score)
        if trainer.should_stop:self.reason='early_stopping'
        if self.total_elapsed()>=self.budget:self.reason='orcamento_tempo';trainer.should_stop=True
        if epoch>=self.tc.epochs:self.reason='maximo_epocas'
        slot=epoch%2;folder=self.output/'retomada'/f'slot_{slot}';folder.mkdir(parents=True,exist_ok=True)
        row={'epoch':epoch,'map50_95':score,'map50':values.get('val/ema_mAP_50',values.get('val/mAP_50')),
          'best_epoch':self.best['epoch'],'loss':{k:v for k,v in values.items() if 'loss' in k},'global_step':trainer.global_step,
          'train_batches':int(trainer.num_training_batches),'seconds_before_persistence':time.monotonic()-self.epoch_start}
        self.history=[r for r in self.history if r['epoch']<epoch]+[row]
        checkpoint=folder/'estado.ckpt';tmp=checkpoint.with_suffix('.tmp');trainer.save_checkpoint(tmp);tmp.replace(checkpoint)
        source_best=self.output/self.best['weight'];snapshot=folder/'melhor_ema.pth';shutil.copyfile(source_best,snapshot)
        c.save(folder/'melhor_24.json',self.best)
        files={n:c.sha(folder/n) for n in ['estado.ckpt','melhor_ema.pth','melhor_24.json']}
        for name in files:self.store.put(folder/name,self.remote+f'/retomada/slot_{slot}/'+name,files[name])
        row['seconds_with_persistence']=time.monotonic()-self.epoch_start
        c.save(self.output/'historico_24.json',self.history);shutil.copyfile(self.output/'historico_24.json',folder/'historico_24.json')
        files['historico_24.json']=c.sha(folder/'historico_24.json')
        self.store.put(folder/'historico_24.json',self.remote+f'/retomada/slot_{slot}/historico_24.json',files['historico_24.json'])
        meta={'context':self.context,'epoch':epoch,'elapsed_seconds':self.total_elapsed(),'budget_seconds':self.budget,
          'files':files,'best':self.best,'reason':self.reason,'global_step':trainer.global_step}
        c.save(folder/'commit.json',meta);self.store.put(folder/'commit.json',self.remote+f'/retomada/slot_{slot}/commit.json')
        self.latest=meta
        if len(self.history)>=c.SPEC['pilot_epochs']:
            durations=[r['seconds_with_persistence'] for r in self.history[:c.SPEC['pilot_epochs']]]
            median=float(sorted(durations)[len(durations)//2]);remaining=max(0,self.budget-self.total_elapsed())
            proof={'epochs_measured':len(durations),'seconds_per_epoch_with_backup':durations,'median_seconds':median,
             'estimated_additional_epochs':int(remaining/median) if median else 0,'remaining_training_minutes':remaining/60,
             'includes_validation_and_checkpoint':True,'continues_same_training':True,'not_a_guarantee':True}
            c.save(self.output/'prova_tempo_24.json',proof)
            self.store.put(self.output/'prova_tempo_24.json',self.remote+'/prova_tempo_24.json')
        if self.total_elapsed()>=self.budget:self.reason='orcamento_tempo';trainer.should_stop=True
        print(f'Época {epoch}/{self.tc.epochs}: mAP50–95 VAL={score:.4f}; melhor={self.best["epoch"]}; '
              f'{row["seconds_with_persistence"]/60:.1f} min com backup; orçamento usado={self.total_elapsed()/60:.1f} min.',flush=True)

def restore(output,store,remote,context):
    output=Path(output);valid=[];existing=False
    for slot in [0,1]:
        rel=remote+f'/retomada/slot_{slot}/commit.json'
        if not store.path(rel).is_file():continue
        existing=True
        try:
            meta=c.read(store.path(rel))
            if meta['context']!=context:raise ValueError('Contexto diferente no slot.')
            valid.append((meta,slot))
        except (ValueError,KeyError,OSError):continue
    for meta,slot in sorted(valid,key=lambda x:x[0]['epoch'],reverse=True):
        try:
            folder=output/'retomada'/f'slot_{slot}';folder.mkdir(parents=True,exist_ok=True)
            for name,digest in meta['files'].items():store.fetch(remote+f'/retomada/slot_{slot}/'+name,folder/name,digest)
            best=folder/'melhor_ema.pth'
            if c.sha(best)!=meta['best']['weight_sha256']:raise ValueError('Melhor peso do slot difere do ponteiro.')
            (output/'treino').mkdir(exist_ok=True)
            shutil.copyfile(best,output/'treino/melhor_ema.pth')
            for name in ['melhor_24.json','historico_24.json']:shutil.copyfile(folder/name,output/name)
            return meta,folder/'estado.ckpt'
        except (ValueError,KeyError,OSError):continue
    if existing:raise RuntimeError('Há registros de retomada, mas nenhum slot completo foi validado; treino anterior preservado.')
    return None,None

def train(job,store,device='cuda',qa_overrides=None):
    output=job['output'];remote=job['remote'];context=job['context'];spec=job['spec'];prior,resume=restore(output,store,remote,context)
    if prior and (prior.get('reason') in {'early_stopping','maximo_epocas','orcamento_tempo'} or prior['elapsed_seconds']>=prior['budget_seconds']):
        result={'completed':True,'epochs':prior['epoch'],'reason':prior.get('reason') or 'orcamento_tempo','reused':True,
          'elapsed_training_seconds':prior['elapsed_seconds']};c.save(output/'execucao_24.json',result);return result
    seed_everything(spec['seed'],workers=True)
    mc=RFDETRSmallConfig(num_classes=12,resolution=spec['resolution'],pretrain_weights=str(job['initial']))
    kwargs=dict(dataset_file='yolo',dataset_dir=str(job['base']),output_dir=str(output/'treino'),epochs=spec['epochs'],
      batch_size=spec['batch_size'],grad_accum_steps=spec['effective_batch']//spec['batch_size'],eval_batch_size=spec['batch_size'],
      lr=spec['lr'],lr_encoder=spec['lr_encoder'],weight_decay=spec['weight_decay'],num_workers=spec['num_workers'],
      seed=spec['seed'],amp_dtype=spec['amp_dtype'] if device!='cpu' else None,use_ema=True,eval_base_model=False,
      early_stopping=True,early_stopping_patience=spec['early_stopping_patience'],early_stopping_min_delta=spec['early_stopping_min_delta'],
      early_stopping_use_ema=True,checkpoint_interval=80,best_model_metric='map',run_test=False,eval_interval=1,
      class_names=c.NAMES,eval_max_dets=300,tensorboard=False,wandb=False,mlflow=False,progress_bar='tqdm',
      multi_scale=False,expanded_scales=False,save_dataset_grids=False,eval_backend='faster_coco_eval',
      lr_scheduler='cosine',warmup_epochs=1.0,lr_scheduler_kwargs={'min_factor':.05},pack_targets=False,num_sanity_val_steps=0)
    if qa_overrides:kwargs.update(qa_overrides)
    tc=TrainConfig(**kwargs)
    module=RFDETRModelModule(mc,tc)
    choices_path=output/'escolhas_execucao_24.json'
    if store.path(remote+'/escolhas_execucao_24.json').is_file():
        store.fetch(remote+'/escolhas_execucao_24.json',choices_path);choices=c.read(choices_path)
        if choices['context']!=context:raise ValueError('Escolhas de memória pertencem a outro contexto.')
        tc.batch_size=choices['batch_size'];tc.grad_accum_steps=choices['grad_accum_steps'];tc.eval_batch_size=choices['batch_size']
    else:
        batch=spec['batch_size'];probe=None
        if device!='cpu':
            module.to('cuda');tc.auto_batch_max_targets_per_image=max(len(r['gt']) for r in job['train'])
            probe=resolve_auto_batch_config(SimpleNamespace(model=module.model,device=torch.device('cuda')),mc,tc,max_micro_batch=8,devices=1)
            batch=max(v for v in [1,2,4] if v<=max(1,probe.safe_micro_batch));module.zero_grad(set_to_none=True)
            module.to('cpu');gc.collect();torch.cuda.empty_cache()
        tc.batch_size=batch;tc.grad_accum_steps=spec['effective_batch']//batch;tc.eval_batch_size=batch
        choices={'context':context,'batch_size':batch,'grad_accum_steps':tc.grad_accum_steps,
          'effective_batch':batch*tc.grad_accum_steps,'amp_dtype':tc.amp_dtype,'resolution':spec['resolution'],
          'probe_safe_micro_batch':probe.safe_micro_batch if probe else None,'probe_does_not_update_weights':True}
        c.save(choices_path,choices);store.put(choices_path,remote+'/escolhas_execucao_24.json')
    c.save(output/'configuracao_rf24.json',{'context':context,'model_config':mc.model_dump(),'train_config':tc.model_dump()})
    available=min(spec['training_minutes']*60,max(60,spec['session_minutes']*60-(time.monotonic()-job.get('gpu_start',time.monotonic()))-spec['evaluation_reserve_minutes']*60))
    callback=EpochPersistence(output,store,remote,context,mc,tc,prior,available)
    trainer=build_trainer(tc,mc,accelerator='cpu' if device=='cpu' else 'gpu',devices=1,enable_model_summary=False)
    # O armazenamento próprio guarda somente dois estados completos. EMA, avaliador e EarlyStopping permanecem nativos.
    trainer.callbacks=[cb for cb in trainer.callbacks if not isinstance(cb,ModelCheckpoint)]+[callback]
    data=RFDETRDataModule(mc,tc)
    if callback.total_elapsed()>=callback.budget:raise RuntimeError('Orçamento já esgotado sem resultado final válido.')
    try:
        seed_everything(spec['seed'],workers=True)
        trainer.fit(module,datamodule=data,ckpt_path=str(resume) if resume else None)
        if callback.latest is None:raise RuntimeError('Nenhuma época completa foi persistida.')
        result={'completed':True,'epochs':callback.latest['epoch'],'reason':callback.reason or 'maximo_epocas',
          'resumed_from':prior['epoch'] if prior else None,'elapsed_training_seconds':callback.total_elapsed(),
          'budget_seconds':callback.budget,'soft_budget_epoch_boundary':True}
        c.save(output/'execucao_24.json',result);store.put(output/'execucao_24.json',remote+'/execucao_24.json');return result
    finally:
        del trainer,module,data;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
