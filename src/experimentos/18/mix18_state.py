"""Recuperação por braço completo; tentativa incompleta reinicia do ponto comum."""
from __future__ import annotations
import json,zipfile
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import diag11_core as core
import mix18_protocol as p16

def safe_child(root,relative):
    q=PurePosixPath(relative)
    if q.is_absolute() or '..' in q.parts or '\\' in relative:raise ValueError('Caminho de recuperação inválido.')
    root=Path(root).resolve();path=root/relative
    if not path.resolve().is_relative_to(root):raise ValueError('Caminho fora da execução.')
    return path

def new_attempt(output,arm):
    if arm not in p16.ARMS:raise ValueError('Braço desconhecido.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    path=Path(output)/'bracos'/arm/'tentativas'/stamp;path.mkdir(parents=True)
    core.save_json(path/'tentativa_iniciada.json',{'arm':arm,'started_at':stamp,
                    'resume_mode':'reinicio_do_checkpoint_inicial; sem carregar otimizador ou EMA de tentativa interrompida'})
    return path

def validate_budget(budget,spec):
    count=spec['epochs']*spec['batches']
    if budget['updates']!=count or budget['batches_seen']!=count:raise ValueError('Orçamento final divergente.')
    expected={'home':count*(spec['batch']-spec['oi_per_batch']),'openimages':count*spec['oi_per_batch']}
    if budget['source_views']!=expected or budget['amp'] or budget['accumulate']!=1:raise ValueError('Exposição/precisão/acumulação divergente.')

def seal(output,arm,attempt,context_key,entries,budget,initial_state,spec):
    output=Path(output);attempt=Path(attempt)
    validate_budget(budget,spec)
    if len(entries)!=spec['epochs'] or {e['epoch'] for e in entries}!=set(range(1,spec['epochs']+1)):
        raise ValueError('Checkpoints do braço incompletos.')
    core.save_json(attempt/'checkpoints.json',entries);core.save_json(attempt/'orcamento_final.json',budget)
    files={str(f.relative_to(attempt)):core.sha256_file(f) for f in sorted(attempt.rglob('*')) if f.is_file() and not f.name.endswith('.tmp')}
    if not files:raise ValueError('Tentativa vazia.')
    obj={'concluido':True,'arm':arm,'context_key':context_key,'attempt_relative':str(attempt.relative_to(output)),
         'files':files,'entries_sha256':p16.digest(entries),'initial_state_sha256':initial_state,
         'budget':budget,'completed_at':datetime.now(timezone.utc).isoformat()}
    core.save_json(output/'bracos'/arm/'concluido.json',obj)
    return obj

def load_completed(output,arm,context_key,initial_state,spec,planned):
    output=Path(output);marker=output/'bracos'/arm/'concluido.json'
    if not marker.is_file():return None
    try:
        obj=json.loads(marker.read_text())
        if not obj['concluido'] or obj['arm']!=arm or obj['context_key']!=context_key or obj['initial_state_sha256']!=initial_state:
            raise ValueError('Contexto/inicialização do braço incompatível.')
        attempt=safe_child(output,obj['attempt_relative'])
        if attempt.parent.parent!=(output/'bracos'/arm).resolve():raise ValueError('Tentativa fora do braço esperado.')
        for rel,sha in obj['files'].items():
            f=safe_child(attempt,rel)
            if not f.is_file() or core.sha256_file(f)!=sha:raise ValueError('Arquivo concluído ausente ou alterado: '+rel)
        entries=json.loads((attempt/'checkpoints.json').read_text())
        if p16.digest(entries)!=obj['entries_sha256']:raise ValueError('Entradas de checkpoints alteradas.')
        if len(entries)!=spec['epochs'] or {e['epoch'] for e in entries}!=set(range(1,spec['epochs']+1)):
            raise ValueError('Lista de checkpoints concluídos incompleta.')
        budget=json.loads((attempt/'orcamento_final.json').read_text());validate_budget(budget,spec)
        if budget!=obj['budget']:raise ValueError('Orçamento diverge do marcador de conclusão.')
        logs=[json.loads(line) for line in (attempt/'auditoria_batches.jsonl').read_text().splitlines() if line.strip()]
        if len(logs)!=len(planned):raise ValueError('Diário de batches incompleto.')
        for row,p in zip(logs,planned):
            if any(row[k]!=p[k] for k in ['step','epoch','ids','input_signature']):raise ValueError('Cronograma concluído incompatível.')
            if row['gradient_audit']['unknown_negative_max']!=0:raise ValueError('Auditoria de gradiente inconsistente.')
        evaluations={}
        for e in entries:
            weight=safe_child(output,e['weight_relative'])
            if core.sha256_file(weight)!=e['weight_sha256']:raise ValueError('Peso de checkpoint alterado.')
            snapshot=safe_child(output,e['snapshot_relative'])
            evaluations[e['id']]=p16.load_gz(snapshot)
        return {'marker':obj,'attempt':attempt,'entries':entries,'budget':budget,'logs':logs,'evaluations':evaluations}
    except (KeyError,TypeError,ValueError,OSError,EOFError) as exc:
        core.save_json(output/'recuperacao'/(arm+'_rejeitado.json'),{'arm':arm,'motivo':str(exc)[:500],
                       'acao':'reiniciar braço do checkpoint inicial; resultado incompleto não entra na seleção'})
        print('Resultado incompleto/incompatível: o braço será reiniciado:',arm,flush=True)
        return None

def package(output):
    output=Path(output);archive=output/'pacote_experimento_18.zip';tmp=archive.with_name(archive.name+'.tmp')
    selected=[]
    for arm in p16.ARMS:
        marker=output/'bracos'/arm/'concluido.json'
        if marker.is_file():
            try:selected.append(json.loads(marker.read_text())['attempt_relative'])
            except (ValueError,KeyError):pass
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in sorted(output.rglob('*')):
            if not f.is_file() or f in {archive,tmp} or f.name.endswith('.tmp') or f.suffix in {'.pt','.pkl'}:continue
            rel=f.relative_to(output)
            if f.name=='predicoes_conf_0p001.jsonl.gz':continue
            if rel.parts[0]=='bracos' and 'tentativas' in rel.parts:
                attempt=str(Path(*rel.parts[:4]))
                if attempt not in selected:
                    if f.name in {'tentativa_iniciada.json','tentativa_interrompida.json'}:z.write(f,rel)
                    continue
            z.write(f,rel)
    tmp.replace(archive)
    return archive
