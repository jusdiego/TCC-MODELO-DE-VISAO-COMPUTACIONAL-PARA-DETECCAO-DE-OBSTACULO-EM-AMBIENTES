"""Bootstrap pareado dos grupos, com denominadores indefinidos preservados."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import diag11_core as core
from final20_data import MODELS,digest,save_gz

METRICS=('precision','recall','f1','taxa_alta_sem_alerta')
SCOPES=('global',)+tuple(core.NAMES)

def make_groups(records,output=None):
    ordered=sorted(records,key=lambda r:r['id'])
    if len({r['id'] for r in ordered})!=len(ordered):raise ValueError('IDs repetidos na formação dos grupos.')
    parent=list(range(len(ordered)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    links=[]
    for i,a in enumerate(ordered):
        for j in range(i):
            b=ordered[j];distance=(int(a['phash'])^int(b['phash'])).bit_count()
            if a['sha256']==b['sha256'] or distance<=4:
                union(i,j);links.append({'a':a['id'],'b':b['id'],'bytes_iguais':a['sha256']==b['sha256'],'distancia_phash':distance})
    buckets=defaultdict(list)
    for i,r in enumerate(ordered):buckets[find(i)].append(r['id'])
    groups=sorted((sorted(ids) for ids in buckets.values()),key=lambda ids:ids[0])
    mapping={iid:digest(ids)[:24] for ids in groups for iid in ids}
    report={'metodo':'SHA-256 igual ou pHash64 Hamming <=4; componentes transitivos',
        'phash_origem':'valores congelados no manifesto autenticado, antes desta avaliação',
        'imagens':len(ordered),'grupos':len(groups),'distribuicao_tamanhos':dict(sorted(Counter(map(len,groups)).items())),
        'maior_grupo':max(map(len,groups),default=0),'componentes':[{'grupo':digest(ids)[:24],'ids':ids} for ids in groups],
        'ligacoes':links,'grupos_sha256':digest(groups)}
    if output is not None:
        output=Path(output);core.save_json(output/'grupos_bootstrap_20.json',report)
        core.save_csv(output/'grupos_por_imagem.csv',[{'id':r['id'],'grupo':mapping[r['id']],'sha256':r['sha256'],'phash':f'{r["phash"]:016x}'} for r in ordered])
    return groups,mapping,report

def rates(counts):
    c=np.asarray(counts,dtype=float)
    tp,fn,fp,high_missing,high_total=[c[...,i] for i in range(5)]
    denominators=[tp+fp,tp+fn,2*tp+fp+fn,high_total]
    numerators=[tp,tp,2*tp,high_missing]
    result=[]
    for n,d in zip(numerators,denominators):
        result.append(np.divide(n,d,out=np.full_like(d,np.nan,dtype=float),where=d>0))
    return np.stack(result,axis=-1)

def build_counts(records,evaluations):
    ids=[r['id'] for r in records];positions={iid:i for i,iid in enumerate(ids)}
    counts=np.zeros((2,len(records),len(SCOPES),5),dtype=np.int64)
    supports=np.zeros(len(SCOPES),dtype=np.int64)
    for r in records:
        for g in r['gt']:
            if r['known_mask']&(1<<g['classe_id']):supports[0]+=1;supports[1+g['classe_id']]+=1
    for mi,model in enumerate(MODELS):
        evaluated=evaluations[model]
        if {v['id'] for v in evaluated['images']}!=set(ids) or len(evaluated['images'])!=len(ids):raise ValueError('Amostras diferentes no pareamento.')
        for event in evaluated['events']:
            if event['id'] not in positions:raise ValueError('Evento fora da reserva.')
            vector=np.asarray([event['ha_alerta_ref'] and event['ha_alerta_pred'],
                event['ha_alerta_ref'] and not event['ha_alerta_pred'],
                not event['ha_alerta_ref'] and event['ha_alerta_pred'],
                event['prioridade_ref']=='alto' and not event['ha_alerta_pred'],event['prioridade_ref']=='alto'],dtype=np.int64)
            i=positions[event['id']];counts[mi,i,0]+=vector;counts[mi,i,1+event['classe_id']]+=vector
        point=counts[mi,:,0].sum(axis=0);s=evaluated['summary']
        if list(point)!=[s['alertas_por_objeto']['tp'],s['alertas_por_objeto']['fn'],s['alertas_por_objeto']['fp'],s['alta_sem_alerta'],s['alta_ref']]:
            raise ValueError('Estatísticas de reamostragem divergem das métricas.')
    if not np.array_equal(counts[0,:,:,0]+counts[0,:,:,1],counts[1,:,:,0]+counts[1,:,:,1]) or not np.array_equal(counts[0,:,:,4],counts[1,:,:,4]):
        raise ValueError('Os dois modelos não usam os mesmos denominadores GT.')
    return counts,supports

def bootstrap(records,evaluations,groups,plan,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    counts,supports=build_counts(records,evaluations)
    positions={r['id']:i for i,r in enumerate(records)}
    flat=[iid for group in groups for iid in group]
    if set(flat)!=set(positions) or len(flat)!=len(positions) or any(not g for g in groups):raise ValueError('Grupos não particionam a reserva.')
    grouped=np.stack([counts[:,[positions[iid] for iid in group]].sum(axis=1) for group in groups],axis=0)
    repetitions=plan['confidence_intervals']['repetitions'];seed=plan['confidence_intervals']['seed']
    rng=np.random.default_rng(seed);g=len(groups)
    weights=np.empty((repetitions,g),dtype=np.int32)
    for i in range(repetitions):weights[i]=np.bincount(rng.integers(0,g,size=g),minlength=g)
    sampled=np.tensordot(weights,grouped,axes=(1,0)) # réplica, modelo, escopo, contagens
    sampled_rates=rates(sampled);points=rates(counts.sum(axis=1))
    differences=sampled_rates[:,1]-sampled_rates[:,0]
    rows=[]
    for si,scope in enumerate(SCOPES):
        total_counts=counts[:,:,si].sum(axis=1)
        for name,mi in [(MODELS[0],0),(MODELS[1],1),('delta_candidato10_menos_baseline',None)]:
            for ki,metric in enumerate(METRICS):
                values=sampled_rates[:,mi,si,ki] if mi is not None else differences[:,si,ki]
                point=points[mi,si,ki] if mi is not None else points[1,si,ki]-points[0,si,ki]
                valid=values[np.isfinite(values)]
                lower,upper=np.percentile(valid,[2.5,97.5]) if len(valid) else (None,None)
                n,d=(None,None)
                if mi is not None:
                    tp,fn,fp,hm,ht=map(int,total_counts[mi]);n=[tp,tp,2*tp,hm][ki];d=[tp+fp,tp+fn,2*tp+fp+fn,ht][ki]
                rows.append({'escopo':scope,'modelo_ou_comparacao':name,'metrica':metric,
                    'estimativa':float(point) if np.isfinite(point) else None,
                    'ic95_inferior':float(lower) if lower is not None else None,
                    'ic95_superior':float(upper) if upper is not None else None,
                    'replicas_validas':len(valid),'replicas_previstas':repetitions,
                    'numerador':n,'denominador':d,'gt_avaliaveis':int(supports[si]),
                    'suporte_reduzido':bool(supports[si]<30),
                    'estimando_primario':scope=='global' and metric=='recall' and name in {MODELS[0],'delta_candidato10_menos_baseline'},
                    'unidade':'proporcao' if mi is not None else 'diferenca_de_proporcoes'})
    core.save_csv(output/'intervalos_confianca_20.csv',rows)
    core.save_json(output/'intervalos_confianca_20.json',rows)
    save_gz(output/'replicas_bootstrap_20.json.gz',{'modelos':MODELS,'escopos':SCOPES,'metricas':METRICS,
        'valores':sampled_rates.tolist(),'delta_candidato10_menos_baseline':differences.tolist()})
    # NPZ permite reproduzir cada estimativa usando as mesmas multiplicidades de grupos.
    temp=output/'reamostragem_grupos_20.tmp.npz'
    np.savez_compressed(temp,multiplicidades=weights,contagens_grupos=grouped,contagens_por_imagem=counts)
    temp.replace(output/'reamostragem_grupos_20.npz')
    report={'metodo':plan['confidence_intervals'],'imagens':len(records),'grupos':g,'pareado':True,
        'metricas':list(METRICS),'escopos':list(SCOPES),'total_estimativas':len(rows),
        'replicas_iguais_ambos_modelos':True,'denominadores_zero_convertidos_em_zero':False,
        'ordem_grupos_sha256':digest(groups),'rng':'numpy.random.default_rng; amostra de G grupos com reposição por réplica',
        'analise':'Razões de somas dos objetos; sem média das métricas por imagem.',
        'limites':['Intervalos por classe e métricas secundárias são descritivos; sem ajuste por múltiplas comparações.',
                  'Grupos pequenos e poucos objetos podem produzir intervalos instáveis ou degenerados.']}
    core.save_json(output/'bootstrap_resumo_20.json',report)
    return rows,report
