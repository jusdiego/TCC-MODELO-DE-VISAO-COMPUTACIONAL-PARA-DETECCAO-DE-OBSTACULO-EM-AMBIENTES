"""Treino longo, amostragem de classes e checkpoint retomável por época."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import gc,json,math,shutil,time
import random
from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.data import Sampler
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model
import diag11_core as core
from exp10_partial import TrainerOpenImages,LoaderFinito,export_standard_checkpoint
from learn21_protocol import SPEC,digest
import learn21_eval as evaluation

@contextmanager
def isolated_validation():
    py=random.getstate();np_state=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    threads=torch.get_num_threads();det=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn=(torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    try:yield
    finally:
        random.setstate(py);np.random.set_state(np_state);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        torch.set_num_threads(threads);torch.use_deterministic_algorithms(det,warn_only=warn)
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=cudnn

class ClassSourceSampler(Sampler):
    def __init__(self,home,oi,classes,batch,home_per_batch,seed=42,uniform_fraction=.75):
        self.home=list(home);self.oi=list(oi);self.classes=classes;self.batch=batch;self.nh=home_per_batch;self.no=batch-home_per_batch
        if not self.home or not self.oi or self.no<=0 or self.nh<=0:raise ValueError('Amostragem exige duas fontes.')
        self.seed=seed;self.uniform=uniform_fraction;self.epoch=0;self.batches=math.ceil(len(oi)/self.no)
        self.by_class={c:[i for i in oi if c in classes[i]] for c in range(12)}
        self.available=[c for c,v in self.by_class.items() if v]
        if not self.available:raise ValueError('Os rótulos processados de Open Images ficaram sem caixas válidas.')
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __len__(self):return self.batches*self.batch
    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]));needed=self.batches*self.no
        uniform=[]
        while len(uniform)<needed:uniform.extend(rng.permutation(self.oi).tolist())
        oi=[]
        for i in range(needed):
            if rng.random()<self.uniform:oi.append(uniform[i])
            else:oi.append(int(rng.choice(self.by_class[int(rng.choice(self.available))])))
        home=[]
        while len(home)<self.batches*self.nh:home.extend(rng.permutation(self.home).tolist())
        for b in range(self.batches):
            batch=home[b*self.nh:(b+1)*self.nh]+oi[b*self.no:(b+1)*self.no];rng.shuffle(batch);yield from batch

class DevValidator:
    def __init__(self,trainer):
        self.trainer=trainer;self.args=deepcopy(trainer.args)
        self.metrics=SimpleNamespace(keys=['dev/precision','dev/recall','dev/f1','dev/threshold','dev/target_met'])
    def __call__(self,trainer):
        # A cópia impede que fusão/predict altere o modelo ou a EMA do treino.
        with isolated_validation():
            wrapper=YOLO('yolov8s.yaml');wrapper.model=deepcopy(trainer.ema.ema).float().eval();wrapper.predictor=None;wrapper.task='detect'
            try:selected,summaries,profiles=evaluation.evaluate(wrapper,trainer.dev_cohorts,trainer.device,trainer.spec)
            finally:del wrapper;gc.collect()
        trainer.pending={'selected':selected,'summaries':summaries,'profiles':profiles}
        return {'dev/precision':selected['detector']['precision'] or 0.,'dev/recall':selected['detector']['recall'] or 0.,
            'dev/f1':selected['detector']['f1'] or 0.,'dev/threshold':selected['threshold'],'dev/target_met':int(selected['target_met']),'fitness':selected['fitness']}

class LongTrainer(TrainerOpenImages):
    spec=SPEC;dev_cohorts=None;output=None;context_key=None;backup_callback=None
    def __init__(self,*args,**kwargs):
        self.session_started=time.monotonic();self.epoch_started=None;self.sources=Counter();self.unique=set();self.updates_seen=0
        self.pending=None;self.pause_reason=None
        super().__init__(*args,**kwargs)
        path=Path(self.output)/'historico_treino_21.json'
        self.history=json.loads(path.read_text()) if path.is_file() else []
        self.add_callback('on_train_epoch_start',self.begin_epoch)
        self.add_callback('on_model_save',self.commit_epoch)
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        if batch_size!=self.spec['batch'] or self.world_size>1:raise ValueError('Batch ou número de GPUs alterado.')
        for key in ['mosaic','mixup','cutmix','copy_paste']:
            if float(getattr(self.args,key,0))!=0:raise ValueError('Mistura de imagens incompatível com a cobertura parcial.')
        dataset=self.build_dataset(dataset_path,mode,batch_size);home=[];oi=[];classes={}
        for i,path in enumerate(dataset.im_files):
            record=self.source_records.get(Path(path).resolve().as_posix())
            if not record or record['role']!='train':raise ValueError('Loader recebeu imagem fora do TRAIN.')
            (home if record['source']=='home' else oi).append(i)
            # Rótulos efetivamente processados pelo YOLO, incluindo seus filtros.
            classes[i]=set(int(x) for x in dataset.labels[i]['cls'].reshape(-1))
        sampler=ClassSourceSampler(home,oi,classes,batch_size,self.spec['home_per_batch'],self.spec['seed'],self.spec['oi_uniform_fraction'])
        return LoaderFinito(dataset,batch_size=batch_size,sampler=sampler,num_workers=self.args.workers,
            pin_memory=self.device.type=='cuda',collate_fn=dataset.collate_fn,drop_last=False,
            generator=torch.Generator().manual_seed(self.spec['seed']))
    def get_validator(self):
        self.loss_names=('box_loss','cls_loss','dfl_loss');return DevValidator(self)
    def begin_epoch(self,trainer):
        self.train_loader.sampler.set_epoch(self.epoch);self.sources.clear();self.unique.clear();self.epoch_started=time.monotonic()
    def preprocess_batch(self,batch):
        result=super().preprocess_batch(batch)
        for name in batch['im_file']:
            r=self.source_records[Path(name).resolve().as_posix()];self.sources[r['source']]+=1;self.unique.add(r['id'])
        return result
    def optimizer_step(self):
        if not torch.isfinite(self.loss.detach()).all():raise FloatingPointError('Loss não finita; checkpoint anterior preservado.')
        super().optimizer_step();self.updates_seen+=1
    def validate(self):
        metrics=self.validator(self);fitness=metrics.pop('fitness')
        if not math.isfinite(fitness):raise FloatingPointError('Fitness não finito.')
        if self.best_fitness is None or fitness>self.best_fitness:self.best_fitness=fitness
        return metrics,fitness
    def _handle_nan_recovery(self,epoch):
        if self.loss is not None and not torch.isfinite(self.loss.detach()).all():raise FloatingPointError('Treino não finito; retome pelo último checkpoint válido.')
        return False
    def save_model(self):
        if any(not torch.isfinite(v).all() for v in self.ema.ema.state_dict().values() if v.is_floating_point()):
            raise FloatingPointError('EMA não finita; o checkpoint anterior será preservado.')
        return super().save_model()
    def final_eval(self):
        # A rotina padrão remove o optimizer de last.pt. A versão retomável deve
        # permanecer completa, inclusive quando a pausa ocorre por tempo.
        return None
    def commit_epoch(self,trainer):
        if self.pending is None:raise RuntimeError('Época sem avaliação de desenvolvimento.')
        output=Path(self.output);epoch=self.epoch+1
        previous=[r for r in self.history if r['epoch']<epoch]
        best_old=max(previous,key=lambda r:(r['fitness'],-r['epoch'])) if previous else None
        best_now=best_old is None or self.fitness>best_old['fitness']
        best_info=json.loads((output/'melhor_modelo_21.json').read_text()) if (output/'melhor_modelo_21.json').is_file() else None
        if best_now:
            target=output/'pesos'/f'epoca_{epoch:03d}_inferencia.pt';target.parent.mkdir(exist_ok=True)
            export_standard_checkpoint(self.last,target)
            best_info={'epoch':epoch,'fitness':float(self.fitness),'threshold':self.pending['selected']['threshold'],
                'weight_relative':target.relative_to(output).as_posix(),'weight_sha256':core.sha256_file(target),'context_key':self.context_key}
            core.save_json(output/'melhor_modelo_21.json',best_info)
        row={'epoch':epoch,'fitness':float(self.fitness),'selected':self.pending['selected'],
            'home':self.pending['summaries']['home_val_historica'],'exposicoes_fontes':dict(self.sources),
            'imagens_unicas_expostas':len(self.unique),'optimizer_steps_nesta_sessao':self.updates_seen,
            'segundos_epoca_com_validacao':time.monotonic()-self.epoch_started}
        self.history=previous+[row];core.save_json(output/'historico_treino_21.json',self.history)
        best_epoch=max(self.history,key=lambda r:(r['fitness'],-r['epoch']))['epoch']
        if epoch-best_epoch>=self.spec['patience']:self.stop=True;self.pause_reason='early_stopping'
        if time.monotonic()-self.session_started>=self.spec['session_hours']*3600 and epoch<self.spec['epochs'] and self.pause_reason is None:
            self.stop=True;self.pause_reason='limite_tempo_sessao'
        if epoch>=self.spec['epochs']:self.pause_reason='epochs_concluidas'
        slot=epoch%2;folder=output/'retomada';folder.mkdir(exist_ok=True)
        target=folder/f'slot_{slot}.pt';tmp=target.with_suffix('.tmp');shutil.copy2(self.last,tmp);tmp.replace(target)
        history_slot=folder/f'historico_slot_{slot}.json';core.save_json(history_slot,self.history)
        core.save_json(folder/f'slot_{slot}.json',{'epoch':epoch,'sha256':core.sha256_file(target),
            'context_key':self.context_key,'checkpoint':target.name,'best':best_info,'pause_reason':self.pause_reason,
            'history_checkpoint':history_slot.name,'history_sha256':core.sha256_file(history_slot)})
        p=self.pending['selected'];print(f'Época {epoch}/{self.spec["epochs"]}: P={p["detector"]["precision"]}, R={p["detector"]["recall"]}, confiança={p["threshold"]}, meta={p["target_met"]}',flush=True)
        print(f'Tempo desta época com validação: {row["segundos_epoca_com_validacao"]/60:.1f} min. Restante sem parada antecipada: {(self.spec["epochs"]-epoch)*row["segundos_epoca_com_validacao"]/3600:.1f} h.',flush=True)
        if self.backup_callback:self.backup_callback()
        self.pending=None

def resume_candidate(output,context_key):
    valid=[]
    for path in (Path(output)/'retomada').glob('slot_*.json'):
        try:
            meta=json.loads(path.read_text());weight=path.with_name(meta['checkpoint'])
            if meta['context_key']!=context_key:raise ValueError('Contexto de retomada diferente.')
            best=meta['best'];best_weight=Path(output)/best['weight_relative']
            history=path.with_name(meta['history_checkpoint'])
            if (weight.is_file() and core.sha256_file(weight)==meta['sha256']
                    and best['context_key']==context_key and best_weight.is_file()
                    and core.sha256_file(best_weight)==best['weight_sha256']
                    and history.is_file() and core.sha256_file(history)==meta['history_sha256']):valid.append((meta,weight))
        except (ValueError,OSError,KeyError):continue
    return max(valid,key=lambda x:x[0]['epoch']) if valid else None

def train(initial,data_yaml,records,cohorts,output,context_key,device,backup=None,spec=SPEC):
    output=Path(output);output.mkdir(exist_ok=True)
    source_map={Path(r['image']).resolve().as_posix():r for r in records}
    for r in records:
        if r['role']!='train':raise ValueError('Conjunto não autorizado no treinamento.')
    prior=resume_candidate(output,context_key)
    if prior:
        # Uma interrupção durante a publicação da época pode deixar histórico
        # ou ponteiro mais novos que o último slot completo.
        core.save_json(output/'melhor_modelo_21.json',prior[0]['best'])
        hp=output/'retomada'/prior[0]['history_checkpoint']
        core.save_json(output/'historico_treino_21.json',json.loads(hp.read_text()))
    elif list((output/'retomada').glob('slot_*.json')):
        raise RuntimeError('Há registros de retomada, mas nenhum checkpoint completo foi validado. Os arquivos foram preservados.')
    if prior and (prior[0]['epoch']>=spec['epochs'] or prior[0].get('pause_reason')=='early_stopping'):
        return {'status':'treino_concluido_reutilizado','concluido':True,'epochs':prior[0]['epoch']}
    class ConfiguredTrainer(LongTrainer):pass
    ConfiguredTrainer.source_records=source_map;ConfiguredTrainer.spec=spec;ConfiguredTrainer.dev_cohorts=cohorts
    ConfiguredTrainer.output=output;ConfiguredTrainer.context_key=context_key;ConfiguredTrainer.backup_callback=staticmethod(backup) if backup else None
    kwargs={k:spec[k] for k in ['epochs','patience','imgsz','batch','optimizer','lr0','lrf','warmup_epochs','weight_decay','cos_lr','nbs','amp',
        'mosaic','mixup','cutmix','copy_paste','degrees','translate','scale','fliplr','flipud','hsv_h','hsv_s','hsv_v','workers','seed']}
    kwargs.update(data=str(data_yaml),device=device,save_dir=str(output/'treino'),exist_ok=True,project=str(output),name='treino',
        cache=False,rect=False,compile=False,plots=False,save=True,save_period=-1,close_mosaic=0,
        val=True,pretrained=True,channels_last=False,deterministic=True,verbose=False)
    if prior:kwargs['resume']=str(prior[1]);model=YOLO(str(prior[1]))
    else:model=YOLO(str(initial))
    try:
        model.train(trainer=ConfiguredTrainer,**kwargs)
        trainer=model.trainer;paused=trainer.pause_reason=='limite_tempo_sessao'
        return {'status':'pausado_retomavel' if paused else 'treino_concluido','concluido':not paused,
            'epochs':trainer.epoch+1,'motivo':trainer.pause_reason,'retomou_de_epoch':prior[0]['epoch'] if prior else None}
    finally:
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
