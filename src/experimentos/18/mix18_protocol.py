"""Experimento 18: duas proporções, mesmo ponto inicial e orçamento."""
from __future__ import annotations
from collections import Counter
import math
from pathlib import Path
import numpy as np
import control16_protocol as historical
from exp12_protocol import BudgetSampler

VERSION='18-proporcao-fontes-1.0.0'
ARMS=('C12_4','E8_8')
SPEC={k:v for k,v in historical.SPEC.items() if k!='oi_per_batch'}
SPECS={a:{**SPEC,'oi_per_batch':oi} for a,oi in zip(ARMS,(4,8))}
AUGMENTATION_ZERO=historical.AUGMENTATION_ZERO
canonical=historical.canonical
digest=historical.digest
save_gz=historical.save_gz
load_gz=historical.load_gz
filename=historical.filename
logical_order=historical.logical_order
schedule=historical.schedule
safe_judge=historical.safe_judge

def validate_specs(specs=SPECS):
    if tuple(specs)!=ARMS:raise ValueError('Braços diferentes do protocolo.')
    if any({k:v for k,v in specs[a].items() if k!='oi_per_batch'}!=SPEC for a in ARMS):
        raise ValueError('Somente a proporção por fonte pode variar.')
    if [specs[a]['oi_per_batch'] for a in ARMS]!=[4,8]:raise ValueError('Proporção não autorizada.')
    return True

def audit_source_streams(records,specs=SPECS):
    """Compara os fluxos antes da permutação das posições dentro do batch."""
    validate_specs(specs)
    ordered=logical_order(records)
    indices={s:[i for i,r in enumerate(ordered) if r['source']==s] for s in ('home','openimages')}
    checks=[]
    for epoch in range(SPEC['epochs']):
        streams={}
        for arm in ARMS:
            spec=specs[arm]
            sampler=BudgetSampler(indices['home'],indices['openimages'],spec['batch'],spec['batches'],spec['oi_per_batch'],spec['seed'])
            sampler.set_epoch(epoch)
            streams[arm]={s:sampler._stream(indices[s],spec['batches']*(spec['batch']-spec['oi_per_batch'] if s=='home' else spec['oi_per_batch']),j)
                          for j,s in enumerate(('home','openimages'))}
        for source in indices:
            a,b=(streams[arm][source] for arm in ARMS);n=min(len(a),len(b))
            if a[:n]!=b[:n]:raise ValueError('Os fluxos aleatórios por fonte deixaram de ser comuns.')
            checks.append({'epoch':epoch+1,'source':source,'common_prefix':n,
                           'common_prefix_sha256':digest([ordered[i]['id'] for i in a[:n]])})
    return {'fluxos_por_fonte_com_prefixo_comum':True,'antes_permutacao_intrabatch':True,'checks':checks}

def target_counts(record):
    rows=np.array([list(map(float,l.split())) for l in Path(record['label']).read_text(encoding='utf-8-sig').splitlines() if l.strip()],dtype=np.float32).reshape(-1,5)
    # O loader autenticado remove linhas integralmente repetidas; nenhuma classe é removida.
    return dict(Counter(str(int(row[0])) for row in np.unique(rows,axis=0)))

def audit_runs(results,plans,records,specs=SPECS):
    validate_specs(specs)
    reference={r['id']:r for r in records}
    if len(reference)!=len(records):raise ValueError('IDs repetidos no treino.')
    targets={iid:target_counts(r) for iid,r in reference.items()}
    initial=[];summaries=[]
    for arm in ARMS:
        result=results[arm];spec=specs[arm];logs=result['logs'];plan=plans[arm]
        if len(logs)!=spec['epochs']*spec['batches'] or len(logs)!=len(plan):raise ValueError('Orçamento observado incompleto.')
        initial.append(result['marker']['initial_state_sha256'])
        views=Counter();ids=set()
        for row,p in zip(logs,plan):
            if any(row[k]!=p[k] for k in ['step','epoch','ids','input_signature']):raise ValueError('Batch diverge do plano por fonte.')
            counts=Counter(reference[i]['source'] for i in row['ids'])
            expected={'home':spec['batch']-spec['oi_per_batch'],'openimages':spec['oi_per_batch']}
            if counts!=expected or row['source_counts']!=expected:raise ValueError('Proporção efetiva diferente do braço.')
            if row['gradient_audit']['unknown_negative_max']!=0:raise ValueError('Gradiente negativo vazou em classe desconhecida.')
            if len(row['known_masks'])!=spec['batch'] or len(row['gt_per_image_by_class'])!=spec['batch']:
                raise ValueError('Máscaras/alvos do batch incompletos.')
            for j,iid in enumerate(row['ids']):
                r=reference[iid]
                actual={str(k):v for k,v in row['gt_per_image_by_class'][j].items()}
                if row['known_masks'][j]!=r['known_mask'] or actual!=targets[iid]:raise ValueError('Alvos ou cobertura alterados entre fontes.')
                if r['source']=='home' and r['known_mask']!=4095:raise ValueError('Home precisa manter cobertura completa.')
            views.update(counts);ids.update(row['ids'])
        if dict(views)!=result['budget']['source_views']:raise ValueError('Exposições divergentes do orçamento salvo.')
        summaries.append({'arm':arm,'batches':len(logs),'source_views':dict(views),
                          'unique_exposed':dict(Counter(reference[i]['source'] for i in ids))})
    if len(set(initial))!=1:raise ValueError('Pesos iniciais diferentes entre os braços.')
    return {'auditoria_conferida':True,'mesmo_estado_inicial':True,'mesmos_dados_disponiveis':True,
            'rotulos_e_cobertura_originais_preservados':True,'gradiente_negativo_desconhecido_zero':True,
            'mesmo_total_atualizacoes':True,'diferenca_planejada':'quantidade de imagens por fonte em cada batch',
            'mesmos_ids_por_batch':False,'bracos':summaries,
            'fluxos':audit_source_streams(records,specs)}

def select_final(entries,rule):
    assessed=[{**e,'checks':safe_judge(e['home'],e['external'],rule)} for e in entries]
    primary=[e for e in assessed if e['arm'] in ARMS and e['epoch']==SPEC['epochs']]
    if len(primary)!=2 or {e['arm'] for e in primary}!=set(ARMS):raise ValueError('Exigidos os dois finais e03.')
    candidate=next(e for e in primary if e['arm']==ARMS[1])
    eligible=candidate if candidate['checks']['elegivel'] else None
    return {'assessed':assessed,'primary_ids':[e['id'] for e in primary],
            'candidate_id':candidate['id'],'eligible':eligible,
            'status':'E8_8_e03_elegivel_em_validacao' if eligible else 'E8_8_e03_nao_cumpre_seis_criterios',
            'checkpoint_predefinido':'E8_8_e03','controle_predefinido':'C12_4_e03',
            'intermediarios':'diagnostico; sem seleção por melhor época',
            'modelo_substituido':False,'modelo_em_uso_continua_baseline':True}
