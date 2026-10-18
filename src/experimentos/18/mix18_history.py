"""Proveniência dos dois resultados anteriores, sem reutilizar seus pesos ajustados."""
from __future__ import annotations
import json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import diag11_core as core

def read_result(path,expected_sha,member):
    if core.sha256_file(path)!=expected_sha:raise ValueError('Resultado anterior diferente do analisado.')
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Pacote com membros repetidos.')
        for name in z.namelist():core.safe_member(name)
        if z.testzip():raise ValueError('Pacote anterior com erro CRC.')
        return json.loads(z.read(member))

def load(package16,package17,expected,output):
    summary16=read_result(package16,expected['pacote_16'],'resumo_experimento_16.json')
    summary17=read_result(package17,expected['pacote_17'],'resumo_calibracao_17.json')
    if not summary16['concluido'] or summary16['checkpoint_final_elegivel'] is not None:
        raise ValueError('Resultado 16 incompatível com o planejamento.')
    if (not summary17['concluido'] or summary17['eligible_calibration']!=0
        or summary17['distinct_configurations_evaluated_verification']!=0):
        raise ValueError('Resultado 17 incompatível com o planejamento.')
    durations=[]
    with zipfile.ZipFile(package16) as z:
        context=json.loads(z.read('contexto_execucao.json'))
        if context['context_key']!=expected['contexto_16']:raise ValueError('Contexto 16 diferente.')
        entries=json.loads(z.read('checkpoints_avaliados.json'))
        oldcontrol=next(e for e in entries if e['arm']=='C_original' and e['epoch']==3)
        for arm in ['C_original','M_home_unknown7']:
            marker=json.loads(z.read('bracos/'+arm+'/concluido.json'))
            started=json.loads(z.read(marker['attempt_relative']+'/tentativa_iniciada.json'))
            start=datetime.strptime(started['started_at'],'%Y%m%d_%H%M%S_%f').replace(tzinfo=timezone.utc)
            end=datetime.fromisoformat(marker['completed_at'])
            durations.append({'arm_16':arm,'seconds':(end-start).total_seconds(),
                              'includes':'treino, avaliações por época e backups; não mede só GPU'})
    result={'pacote16_sha256':expected['pacote_16'],'pacote17_sha256':expected['pacote_17'],
            'resultado16':summary16['status'],'resultado17':summary17['status'],
            'controle16_e03':oldcontrol,'tempos_16':durations,
            'decisao':'refazer 12/4 e comparar com 8/8 na mesma execução, ambos a partir de C10',
            'estimativa_T4_minutos':[20,40],'tempo_pode_exceder':'downloads e salvamentos lentos, instalação ou nova tentativa'}
    core.save_json(Path(output)/'historico_e_estimativa.json',result)
    return result
