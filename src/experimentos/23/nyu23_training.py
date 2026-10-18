"""YOLOv8s padrão, seleção VAL e retomada completa em dois slots."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import gc,json,math,random,shutil,time
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
import nyu22_common as c
import nyu23_eval as ev

@contextmanager
def isolated():
    py=random.getstate();np_state=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    threads=torch.get_num_threads();det=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled();flags=(torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    try:yield
    finally:
        random.setstate(py);np.random.set_state(np_state);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        torch.set_num_threads(threads);torch.use_deterministic_algorithms(det,warn_only=warn);torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=flags

def export(source,target):
    obj=torch.load(source,map_location='cpu',weights_only=False);model=obj.get('ema') or obj.get('model')
    if type(model) is not DetectionModel or list(model.names.values())!=c.NAMES:raise ValueError('Checkpoint não é DetectionModel padrão com seis classes.')
    model.criterion=None;model=model.float().eval()
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):raise FloatingPointError('Peso de inferência não finito.')
    obj.update(model=model,ema=None,optimizer=None);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);torch.save(obj,target)

class Validator:
    def __init__(self,trainer):
        self.args=deepcopy(trainer.args);self.metrics=SimpleNamespace(keys=['dev/precision','dev/recall','dev/f1','dev/threshold','dev/target_met'])
    def __call__(self,trainer):
        with isolated():
            wrapper=YOLO('yolov8s.yaml');wrapper.model=deepcopy(trainer.ema.ema).float().eval();wrapper.predictor=None;wrapper.task='detect'
            try:selected,_=ev.evaluate_development(wrapper,trainer.rows,trainer.device,trainer.spec)
            finally:del wrapper;gc.collect()
        trainer.pending=selected
        return {'dev/precision':selected['detector']['precision'] or 0.,'dev/recall':selected['detector']['recall'] or 0.,
            'dev/f1':selected['detector']['f1'] or 0.,'dev/threshold':selected['threshold'],'dev/target_met':int(selected['target_met']),'fitness':selected['fitness']}

class Trainer(DetectionTrainer):
    spec=c.TRAIN_SPEC;rows=None;output=None;context=None;backup=None
    def __init__(self,*args,**kwargs):
        self.session_started=time.monotonic();self.epoch_started=None;self.pending=None;self.reason=None
        if ultralytics.__version__!='8.4.150':raise RuntimeError('Use ultralytics==8.4.150.')
        super().__init__(*args,**kwargs)
        hp=Path(self.output)/'historico_nyu23.json';self.history=c.read(hp) if hp.is_file() else []
        self.add_callback('on_train_epoch_start',self.begin);self.add_callback('on_model_save',self.commit)
    def get_validator(self):self.loss_names=('box_loss','cls_loss','dfl_loss');return Validator(self)
    def begin(self,trainer):self.epoch_started=time.monotonic()
    def validate(self):
        metrics=self.validator(self);fitness=metrics.pop('fitness')
        if not math.isfinite(fitness):raise FloatingPointError('Fitness não finito.')
        if self.best_fitness is None or fitness>self.best_fitness:self.best_fitness=fitness
        return metrics,fitness
    def save_model(self):
        if any(not torch.isfinite(v).all() for v in self.ema.ema.state_dict().values() if v.is_floating_point()):raise FloatingPointError('EMA não finita; retomada anterior preservada.')
        return super().save_model()
    def final_eval(self):return None  # Não retirar optimizer do checkpoint retomável.
    def commit(self,trainer):
        output=Path(self.output);epoch=self.epoch+1
        previous=[r for r in self.history if r['epoch']<epoch];old=max(previous,key=lambda r:(r['fitness'],-r['epoch'])) if previous else None
        best=c.read(output/'melhor_nyu23.json') if (output/'melhor_nyu23.json').is_file() else None
        if old is None or self.fitness>old['fitness']:
            target=output/'pesos'/f'epoca_{epoch:03d}.pt';export(self.last,target)
            best={'epoch':epoch,'fitness':float(self.fitness),'threshold':self.pending['threshold'],'weight':target.relative_to(output).as_posix(),'weight_sha256':c.sha(target),'context':self.context}
            c.save(output/'melhor_nyu23.json',best)
        row={'epoch':epoch,'fitness':float(self.fitness),'selected':{k:v for k,v in self.pending.items() if k!='image_rows'},'seconds':time.monotonic()-self.epoch_started}
        self.history=previous+[row];c.save(output/'historico_nyu23.json',self.history)
        best_epoch=max(self.history,key=lambda r:(r['fitness'],-r['epoch']))['epoch']
        if epoch-best_epoch>=self.spec['patience']:self.stop=True;self.reason='early_stopping'
        if epoch>=self.spec['epochs']:self.reason='epochs_completed'
        elif self.reason is None and time.monotonic()-self.session_started>=self.spec['session_hours']*3600:self.stop=True;self.reason='session_time_pause'
        slot=epoch%2;folder=output/'retomada';folder.mkdir(exist_ok=True);target=folder/f'slot_{slot}.pt';tmp=target.with_suffix('.tmp');shutil.copy2(self.last,tmp);tmp.replace(target)
        hs=folder/f'historico_slot_{slot}.json';c.save(hs,self.history)
        c.save(folder/f'slot_{slot}.json',{'epoch':epoch,'checkpoint':target.name,'sha256':c.sha(target),'context':self.context,
            'history':hs.name,'history_sha256':c.sha(hs),'best':best,'reason':self.reason})
        det=self.pending['detector'];print(f'Época {epoch}/{self.spec["epochs"]}: P={det["precision"]}, R={det["recall"]}, confiança={self.pending["threshold"]}, meta={self.pending["target_met"]}',flush=True)
        print(f'Época com validação: {row["seconds"]/60:.1f} min. Projeção máxima restante: {(self.spec["epochs"]-epoch)*row["seconds"]/3600:.1f} h.',flush=True)
        if self.backup:self.backup()
        self.pending=None

def local_resume(output,context):
    output=Path(output);options=[]
    for path in (output/'retomada').glob('slot_*.json'):
        try:
            m=c.read(path);weight=path.with_name(m['checkpoint']);hs=path.with_name(m['history']);best=output/m['best']['weight']
            if m['context']!=context:continue
            if all(p.is_file() and c.sha(p)==s for p,s in [(weight,m['sha256']),(hs,m['history_sha256']),(best,m['best']['weight_sha256'])]):options.append((m,weight))
        except (ValueError,KeyError,OSError):continue
    return max(options,key=lambda x:x[0]['epoch']) if options else None

def sync(storage,output,remote):
    output=Path(output)
    try:
        for p in sorted(output.rglob('*')):
            if not p.is_file() or p.suffix in {'.tmp','.part','.zip'} or '__pycache__' in p.parts:continue
            rel=p.relative_to(output).as_posix()
            if rel.startswith('treino/weights/'):continue
            storage.put(p,remote+'/'+rel)
        return True
    except Exception as exc:print('Backup incompleto; saída local preservada:',type(exc).__name__,flush=True);return False

def restore(storage,output,remote,context):
    output=Path(output);source=storage.path(remote)
    if not source.is_dir():return None
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.suffix in {'.pt','.zip','.tmp','.part'} or p.name.endswith('.sha256.json'):continue
        rel=p.relative_to(source).as_posix()
        if rel.startswith('treino/weights/'):continue
        storage.pull(remote+'/'+rel,output/rel)
    options=[]
    for p in (output/'retomada').glob('slot_*.json'):
        try:
            m=c.read(p)
            if m['context']==context:options.append(m)
        except (ValueError,KeyError):continue
    for m in sorted(options,key=lambda x:x['epoch'],reverse=True):
        try:
            storage.pull(remote+'/retomada/'+m['checkpoint'],output/'retomada'/m['checkpoint'],m['sha256'])
            storage.pull(remote+'/retomada/'+m['history'],output/'retomada'/m['history'],m['history_sha256'])
            storage.pull(remote+'/'+m['best']['weight'],output/m['best']['weight'],m['best']['weight_sha256'])
        except Exception as exc:print('Slot incompleto; tentando o anterior:',type(exc).__name__,flush=True);continue
        c.save(output/'historico_nyu23.json',c.read(output/'retomada'/m['history']));c.save(output/'melhor_nyu23.json',m['best']);return m
    if options:raise RuntimeError('Nenhum checkpoint remoto completo passou nas verificações. Os arquivos foram preservados.')
    return None

def train(initial,yaml_file,train_rows,val_rows,output,context,device,backup=None,spec=c.TRAIN_SPEC):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(r['role']!='train' for r in train_rows) or any(r['role']!='val' for r in val_rows):raise ValueError('Papel proibido no treinamento.')
    frozen=output/'decisao_antes_teste_nyu23.json'
    if frozen.is_file():
        m=c.read(frozen)
        if m['context']!=context:raise ValueError('Decisão de outro contexto.')
        return {'completed':True,'epochs':m['epochs'],'status':'completed_frozen_reused'}
    prior=local_resume(output,context)
    if prior:
        c.save(output/'historico_nyu23.json',c.read(output/'retomada'/prior[0]['history']));c.save(output/'melhor_nyu23.json',prior[0]['best'])
        if prior[0]['epoch']>=spec['epochs'] or prior[0]['reason']=='early_stopping':return {'completed':True,'epochs':prior[0]['epoch'],'status':'completed_reused'}
    elif list((output/'retomada').glob('slot_*.json')):raise RuntimeError('Metadados existem, mas não há checkpoint completo validado.')
    class Configured(Trainer):pass
    Configured.spec=spec;Configured.rows=val_rows;Configured.output=output;Configured.context=context;Configured.backup=staticmethod(backup) if backup else None
    keys=['epochs','patience','imgsz','batch','workers','optimizer','lr0','lrf','warmup_epochs','weight_decay','cos_lr','nbs','amp',
          'mosaic','close_mosaic','mixup','cutmix','copy_paste','degrees','translate','scale','fliplr','flipud','hsv_h','hsv_s','hsv_v','seed']
    kwargs={k:spec[k] for k in keys};kwargs.update(data=str(yaml_file),device=device,project=str(output),name='treino',save_dir=str(output/'treino'),
        exist_ok=True,cache=False,compile=False,channels_last=False,deterministic=True,plots=False,save=True,val=True,verbose=False)
    model=YOLO(str(prior[1] if prior else initial))
    if not prior:
        # O remapeamento nativo de linhas da cabeça usa nomes. Estes sinônimos
        # preservam o pré-treino de sofá/mesa; rótulos NYU não são alterados.
        aliases=spec['pretrained_name_aliases']
        model.model.names={i:aliases.get(name,name) for i,name in model.model.names.items()}
    if prior:kwargs['resume']=str(prior[1])
    try:
        model.train(trainer=Configured,**kwargs);t=model.trainer;paused=t.reason=='session_time_pause'
        return {'completed':not paused,'epochs':t.epoch+1,'status':'paused_resumable' if paused else 'completed','reason':t.reason,'resumed_from':prior[0]['epoch'] if prior else None}
    finally:
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
