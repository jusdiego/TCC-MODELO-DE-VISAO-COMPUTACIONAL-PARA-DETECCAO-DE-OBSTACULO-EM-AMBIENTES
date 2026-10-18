"""Execução com backups por época e reaproveitamento apenas de braços completos."""
from __future__ import annotations
import json,time,gc
from pathlib import Path
import torch
from ultralytics import YOLO
import diag11_core as core
from exp10_partial import export_standard_checkpoint
from exp12_training import check_epoch
import mix18_protocol as p16
import mix18_training as tr16
import mix18_eval as ev16
import mix18_state as state

def run_arm(arm,data,initial,initial_state,spec,planned,context_key,cohorts,policy,matcher,environment,
            output,local,device,backup,arguments):
    output=Path(output)
    reused=state.load_completed(output,arm,context_key,initial_state,spec,planned)
    if reused:
        print('Braço completo conferido e reutilizado:',arm,flush=True)
        return reused
    attempt=state.new_attempt(output,arm);entries=[];evaluations={};done=set();start=time.monotonic()
    spec={**spec,'seed':arguments['seed']}
    Trainer=type('Trainer_'+arm,(tr16.SourceMixTrainer,),{
        'source_records':data['records'],'control_spec':spec,'arm':arm,
        'expected_schedule':planned,'attempt_dir':attempt,'expected_initial_state':initial_state})
    model=YOLO(str(initial))
    def after_save(trainer):
        epoch=trainer.epoch+1
        if epoch in done:return
        check_epoch(trainer)
        if len(trainer.batch_log)!=trainer.actual_updates:raise RuntimeError('Diário de batches incompleto.')
        cid=f'{arm}_e{epoch:02d}';weight=attempt/'pesos'/(cid+'.pt');weight.parent.mkdir(exist_ok=True)
        export=export_standard_checkpoint(trainer.last,weight)
        core.save_json(attempt/'exports'/(cid+'.json'),export)
        budget={'arm':arm,'epoch':epoch,'updates':trainer.actual_updates,'batches_seen':trainer.batches_seen,
                'source_views':dict(trainer.source_views),'amp':False,'accumulate':trainer.accumulate}
        core.save_json(attempt/'orcamento'/(cid+'.json'),budget)
        backup('checkpoint salvo: '+cid)
        evaluation_dir=attempt/'avaliacoes'/cid
        evaluated=ev16.evaluate(weight,cid,cohorts,evaluation_dir,policy,matcher,environment,device)
        entry=ev16.entry(cid,arm,epoch,trainer.actual_updates,str(weight.relative_to(output)),core.sha256_file(weight),evaluated)
        entry['snapshot_relative']=str((evaluation_dir/'avaliacao_snapshot.json.gz').relative_to(output))
        entries.append(entry);evaluations[cid]=evaluated;done.add(epoch)
        core.save_json(attempt/'checkpoints.json',entries)
        elapsed=time.monotonic()-start
        print(f'{arm}: época curta {epoch}/{spec["epochs"]}; {trainer.actual_updates}/{spec["epochs"]*spec["batches"]} atualizações; tempo {elapsed/60:.1f} min',flush=True)
        backup('checkpoint avaliado: '+cid)
    model.add_callback('on_model_save',after_save)
    try:
        print('Iniciando',arm,'| atualizações planejadas:',spec['epochs']*spec['batches'],flush=True)
        model.train(trainer=Trainer,data=str(data['yaml']),project=str(Path(local)/'treinos'),
                    name=arm+'_'+attempt.name,**arguments)
        trainer=model.trainer;check_epoch(trainer)
        budget={'arm':arm,'epoch':spec['epochs'],'updates':trainer.actual_updates,'batches_seen':trainer.batches_seen,
                'source_views':dict(trainer.source_views),'amp':False,'accumulate':trainer.accumulate}
        marker=state.seal(output,arm,attempt,context_key,entries,budget,initial_state,spec)
        backup('braço concluído: '+arm)
        result={'marker':marker,'attempt':attempt,'entries':entries,'evaluations':evaluations,
                'budget':budget,'logs':list(trainer.batch_log),'elapsed_seconds':time.monotonic()-start}
        del trainer,model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return result
    except Exception as exc:
        core.save_json(attempt/'tentativa_interrompida.json',{'arm':arm,'concluido':False,'error_type':type(exc).__name__,
            'message':str(exc)[:1200],'last_evaluated_epochs':sorted(done),
            'restart_policy':'reiniciar este braço do candidato 10; nunca carregar o otimizador interrompido'})
        try:backup('tentativa interrompida: '+arm,obrigatorio=False)
        except Exception:pass
        raise
