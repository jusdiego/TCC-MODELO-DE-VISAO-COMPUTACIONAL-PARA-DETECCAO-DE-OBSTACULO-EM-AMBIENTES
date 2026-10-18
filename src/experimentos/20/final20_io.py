"""Pacote final e estado de execução, sem pesos, imagens ou credenciais."""
from __future__ import annotations
from datetime import datetime,timezone
import json,zipfile
from pathlib import Path
import diag11_core as core

def now():return datetime.now(timezone.utc).isoformat()

def state(output,**changes):
    path=Path(output)/'estado_execucao_20.json'
    previous=json.loads(path.read_text()) if path.is_file() else {}
    previous.update(changes);previous['atualizado_em_utc']=now();core.save_json(path,previous)
    return previous

def package(output):
    output=Path(output);target=output/'pacote_avaliacao_final_20.zip';tmp=target.with_name(target.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(output.rglob('*')):
            if not p.is_file() or p in {target,tmp} or p.name=='salvamento_drive_20.json' or p.suffix in {'.pt','.tmp','.part','.jpg','.jpeg','.png','.webp','.bmp'}:continue
            rel=p.relative_to(output)
            if '__pycache__' in rel.parts:continue
            z.write(p,rel.as_posix())
    tmp.replace(target);return target

def manifest(output):
    output=Path(output);excluded={'integridade_arquivos_20.json','pacote_avaliacao_final_20.zip','salvamento_drive_20.json'}
    rows=[]
    for p in sorted(output.rglob('*')):
        if p.is_file() and p.name not in excluded and p.suffix not in {'.pt','.tmp','.part','.jpg','.jpeg','.png','.webp','.bmp'} and '__pycache__' not in p.parts:
            rows.append({'arquivo':p.relative_to(output).as_posix(),'bytes':p.stat().st_size,'sha256':core.sha256_file(p)})
    core.save_json(output/'integridade_arquivos_20.json',{'arquivos':rows,'gerado_em_utc':now()})
    return rows

def backup(drive,output,remote):
    try:ok=drive.backup(output,remote,required=False)
    except Exception:ok=False
    if not ok:print('Cópia no Drive indisponível neste momento; arquivos locais preservados.',flush=True)
    return ok

def failure(output,error,stage,drive=None,remote=None):
    output=Path(output)
    status=state(output,concluido=False,etapa=stage,erro={'tipo':type(error).__name__,'mensagem':str(error)[:1400]})
    core.save_json(output/'execucao_interrompida_20.json',status)
    core.save_json(output/'resumo_avaliacao_final_20.json',{**status,'status':'execucao_interrompida','pacote_parcial':True})
    manifest(output)
    result=package(output)
    if drive and remote:backup(drive,output,remote)
    return result
