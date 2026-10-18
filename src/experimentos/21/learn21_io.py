"""Resultados leves e retomada verificada, preservando dois checkpoints de treino."""
from pathlib import Path
import json,shutil,zipfile
from datetime import datetime,timezone
import diag11_core as core

def backup(drive,output,remote):
    # Checkpoints de origem são transitórios; os dois slots retomáveis e os pesos
    # de inferência publicados são os arquivos usados para recuperação.
    result=drive._call(['copy',str(output),drive.root+'/'+remote,'--exclude','**/*.tmp','--exclude','**/*.part',
        '--exclude','**/__pycache__/**','--exclude','treino/weights/**','--exclude','*.zip'])
    if result.returncode:print('Backup não concluído; os arquivos locais continuam preservados.',flush=True)
    return result.returncode==0

def restore(drive,output,remote,context):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not drive.exists(remote):return None
    result=drive._call(['copy',drive.root+'/'+remote,str(output),'--exclude','**/*.pt','--exclude','*.zip','--exclude','**/*.tmp','--exclude','treino/weights/**'])
    if result.returncode:raise RuntimeError('Não foi possível recuperar os metadados da execução anterior.')
    options=[]
    for p in (output/'retomada').glob('slot_*.json'):
        try:
            meta=json.loads(p.read_text())
            if meta['context_key']==context:options.append((meta,p))
        except (ValueError,KeyError):pass
    for meta,p in sorted(options,key=lambda x:x[0]['epoch'],reverse=True):
        try:
            drive.fetch(remote+'/retomada/'+meta['checkpoint'],p.with_name(meta['checkpoint']),meta['sha256'])
            best=meta['best']
            drive.fetch(remote+'/'+best['weight_relative'],output/best['weight_relative'],best['weight_sha256'])
            history_file=output/'retomada'/meta['history_checkpoint']
            drive.fetch(remote+'/retomada/'+meta['history_checkpoint'],history_file,meta['history_sha256'])
        except Exception as exc:
            print('Checkpoint remoto incompleto; tentando o anterior:',type(exc).__name__,flush=True);continue
        core.save_json(output/'historico_treino_21.json',json.loads(history_file.read_text()))
        core.save_json(output/'melhor_modelo_21.json',best)
        return meta
    if options:raise RuntimeError('Nenhum checkpoint remoto completo passou na verificação; resultados anteriores foram preservados.')
    return None

def package(output):
    output=Path(output);target=output/'pacote_treinamento_21.zip';tmp=target.with_suffix('.tmp')
    files=[];integrity=output/'integridade_pacote_21.json'
    for p in sorted(output.rglob('*')):
        if not p.is_file() or p in {target,tmp,integrity} or p.suffix in {'.tmp','.part','.jpg','.jpeg','.png','.bmp','.webp'}:continue
        rel=p.relative_to(output)
        if '__pycache__' in rel.parts or 'retomada' in rel.parts or rel.as_posix().startswith('treino/weights/'):continue
        if p.suffix=='.pt' and rel.as_posix()!='melhor_modelo.pt':continue
        files.append(p)
    core.save_json(integrity,{'algoritmo':'sha256','arquivos':{p.relative_to(output).as_posix():core.sha256_file(p) for p in files}})
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files+[integrity]:z.write(p,p.relative_to(output).as_posix())
    tmp.replace(target);return target

def failure(output,error,stage,drive=None,remote=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    core.save_json(output/'resumo_treinamento_21.json',{'concluido':False,'status':'execucao_interrompida','etapa':stage,
        'erro':{'tipo':type(error).__name__,'mensagem':str(error)[:1600]},'preenchimento_manual_obrigatorio':False,
        'modelo_substituido':False,'teste_final_novo_consultado':False})
    archive=package(output)
    if drive and remote:
        try:backup(drive,output,remote)
        except Exception:pass
    return archive
