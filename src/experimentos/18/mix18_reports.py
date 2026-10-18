"""Desfecho pareado e03, comparações por classe e pacote compacto."""
from __future__ import annotations
import html,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diag11_core as core
import mix18_protocol as p16

def table(rows):
    if not rows:return '<p>Sem registros.</p>'
    keys=list(rows[0]);out='<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'
    for r in rows:out+='<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>'
    return out+'</table></div>'

def report(output,entries,evaluations,references,rule,pairing,config,budgets):
    output=Path(output);selection=p16.select_final(entries,rule);rows=[]
    for e in selection['assessed']:
        row={k:e[k] for k in ['id','arm','epoch','updates','weight_sha256']}
        for field,prefix in [('home','home'),('external','externa')]:
            s=e[field];row.update({prefix+'_'+k:v for k,v in s['alertas_por_objeto'].items()})
            row[prefix+'_alta_sem_alerta']=s['alta_sem_alerta']
        row.update(e['checks']);row['desfecho_primario']=e['id'] in selection['primary_ids'];rows.append(row)
    core.save_csv(output/'comparacao_checkpoints.csv',rows);core.save_json(output/'selecao_checkpoints.json',selection)
    finals={e['arm']:e for e in selection['assessed'] if e['id'] in selection['primary_ids']}
    c,m=[finals[a] for a in p16.ARMS];delta=[];class_rows=[];byimage=[]
    for cohort,label in [('home_val_historica','home'),('openimages_val_nova','externa')]:
        ca=evaluations[c['id']][cohort];ma=evaluations[m['id']][cohort]
        for metric in ['tp','fn','fp','precision','recall','f1']:
            x=ca['summary']['alertas_por_objeto'][metric];y=ma['summary']['alertas_por_objeto'][metric]
            delta.append({'coorte':cohort,'metrica':metric,'controle_e03':x,'equilibrado_e03':y,
                          'delta_equilibrado_menos_controle':y-x if x is not None and y is not None else None,
                          'unidade':'proporcao; multiplicar delta por 100 para pontos percentuais' if metric in {'precision','recall','f1'} else 'contagem'})
        for metric in ['alta_ref','alta_sem_alerta']:
            x=ca['summary'][metric];y=ma['summary'][metric]
            delta.append({'coorte':cohort,'metrica':metric,'controle_e03':x,'equilibrado_e03':y,
                          'delta_equilibrado_menos_controle':y-x,'unidade':'contagem'})
        cc={r['classe_id']:r for r in ca['classes']};mc={r['classe_id']:r for r in ma['classes']}
        for cid,name in enumerate(core.NAMES):
            row={'coorte':cohort,'classe_id':cid,'classe':name}
            for key in ['gt_avaliaveis','tp','fn','fp','precision','recall','alta_sem_alerta']:
                x,y=cc[cid][key],mc[cid][key]
                row.update({'controle_'+key:x,'equilibrado_'+key:y,'delta_'+key:y-x if x is not None and y is not None else None})
            class_rows.append(row)
        ci={r['id']:r for r in ca['images']};mi={r['id']:r for r in ma['images']}
        if ci.keys()!=mi.keys():raise RuntimeError('Comparação final exige os mesmos IDs.')
        for iid in ci:
            r={'coorte':cohort,'id':iid}
            for key in ['tp_alerta','fn_alerta','fp_alerta','alta_sem_alerta']:
                r.update({'controle_'+key:ci[iid][key],'equilibrado_'+key:mi[iid][key],'delta_'+key:mi[iid][key]-ci[iid][key]})
            byimage.append(r)
    core.save_csv(output/'efeito_pareado_e03.csv',delta)
    core.save_csv(output/'efeito_pareado_e03_por_classe.csv',class_rows)
    core.save_csv(output/'efeito_pareado_e03_por_imagem.csv',byimage)
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    initial=next(e for e in entries if e['arm']=='inicial')
    for col,(field,title) in enumerate([('home','Home VAL histórica'),('external','Validação externa de desenvolvimento')]):
        for arm,color in zip(p16.ARMS,['#2176b5','#d65a2f']):
            seq=[initial]+sorted([e for e in entries if e['arm']==arm],key=lambda e:e['epoch'])
            x=[e['updates'] for e in seq]
            for row,metric in enumerate(['precision','recall','alta_sem_alerta']):
                ys=[e[field][metric] if metric=='alta_sem_alerta' else e[field]['alertas_por_objeto'][metric] for e in seq]
                ys=[None if y is None else y if metric=='alta_sem_alerta' else 100*y for y in ys]
                axes[row,col].plot(x,ys,'o-',label=arm,color=color)
                axes[row,col].grid(alpha=.25);axes[row,col].set_ylabel(metric+(' (%)' if metric!='alta_sem_alerta' else ''))
        axes[0,col].set_title(title);axes[0,col].legend(fontsize=8);axes[2,col].set_xlabel('Atualizações por braço')
    fig.savefig(output/'curvas_validacao.png',dpi=150);plt.close(fig)
    chosen=selection['eligible']
    status=('O candidato equilibrado e03 cumpriu os seis requisitos nas validações: '+chosen['id']+'.' if chosen else
            'O candidato equilibrado e03 não cumpriu os seis requisitos. O baseline continua como referência.')
    summary={'versao':p16.VERSION,'concluido':True,'status':selection['status'],'mensagem':status,
        'checkpoint_final_elegivel':chosen['id'] if chosen else None,'desfecho_primario':'comparacao_e03_vs_e03',
        'atualizacoes_finais_por_braco':budgets[0]['updates'],
        'orcamentos':budgets,'pareamento':pairing,'novos_bracos_concluidos':2,
        'modelo_substituido':False,'modelo_em_uso_continua_baseline':True,
        'rotulos_originais_preservados':True,'projecao_treino_home7_documentada':False,
        'validacao_rotulos_alterados':False,'reserva_futura_extraida':False,'reserva_futura_inferida':False,
        'test_home_inferido':False,'test_oi20k_inferido':False,'preenchimento_manual_obrigatorio':False,
        'limites':['Uma semente e orçamento curto; não prova superioridade geral ou causalidade fora deste controle.',
                  'As duas validações já participaram do desenvolvimento; não são teste final independente.',
                  'Home pode ter diferenças semânticas também na avaliação, preservada igualmente para ambos.',
                  'O candidato final e03 foi pré-fixado; cumprir limites não prova superioridade estatística sobre 12/4.',
                  'Alertas medem concordância com a política V2, sem validação independente de risco físico.',
                  'A origem do treino original do baseline permanece não determinada.'],
        'configuracao':config,'candidato_predefinido':'E8_8_e03','controle_predefinido':'C12_4_e03'}
    core.save_json(output/'resumo_experimento_18.json',summary)
    display_rows=[]
    for row in rows:
        display_rows.append({'Checkpoint':row['id'],'Primário e03':row['desfecho_primario'],
            'P Home':None if row['home_precision'] is None else f"{100*row['home_precision']:.2f}%",
            'R Home':None if row['home_recall'] is None else f"{100*row['home_recall']:.2f}%",
            'Alta omitida Home':row['home_alta_sem_alerta'],
            'P externa':None if row['externa_precision'] is None else f"{100*row['externa_precision']:.2f}%",
            'R externa':None if row['externa_recall'] is None else f"{100*row['externa_recall']:.2f}%",
            'Alta omitida externa':row['externa_alta_sem_alerta'],'Seis critérios':row['elegivel']})
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Experimento 18 — Proporção das fontes</title><style>body{font:16px/1.6 system-ui;background:#eef3f8;color:#152538;margin:0}main{max-width:1450px;margin:auto;padding:30px}.note{padding:20px;background:white;border:1px solid #dce5ee;border-radius:10px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #dce5ee;padding:9px;text-align:left}img{max-width:100%;height:auto}h1{line-height:1.15}</style><main><p>TCC · Visão computacional</p><h1>Experimento 18 — Mistura 12/4 versus 8/8</h1>'''
    page+='<div class="note"><strong>'+html.escape(status)+'</strong><p>Controle C12_4: 12 Home + 4 Open Images por batch. Candidato E8_8: 8 + 8. Ambos preservam os rótulos e a cobertura originais e partem do candidato 10. A comparação principal usa e03 contra e03; o candidato foi fixado antes do treino.</p></div>'
    page+='<h2>Checkpoints nas duas validações</h2>'+table(display_rows)+'<img src="curvas_validacao.png" alt="Precisão, recall e omissões altas nas duas validações">'
    page+='<h2>Efeito pareado no desfecho fixo e03</h2>'+table(delta)
    page+='<h2>Pareamento e orçamento</h2><p>Mesmo conjunto disponível, estado inicial, 342 atualizações e parâmetros. A diferença planejada é a exposição por fonte: 4.104 Home / 1.368 OI no controle, 2.736 / 2.736 no candidato. Os IDs de cada batch diferem conforme a composição. Os fluxos por fonte compartilham prefixos antes da permutação interna. Ver auditoria_fontes.json e os diários de batches.</p>'
    page+='<p>Ver efeito_pareado_e03_por_classe.csv para conferir se os ganhos e perdas se concentram em alguma classe. O atendimento aos limites históricos e o efeito da proporção sobre o controle são resultados distintos; uma semente não demonstra superioridade geral.</p>'
    page+='<h2>Limites</h2><ul>'+''.join('<li>'+html.escape(t)+'</li>' for t in summary['limites'])+'</ul></main></html>'
    (output/'relatorio_experimento_18.html').write_text(page,encoding='utf-8')
    (output/'LEIA_RESULTADO.md').write_text('# Experimento 18\n\n'+status+'\n\nEnvie pacote_experimento_18.zip. O desfecho principal é e03 contra e03; e01/e02 são diagnósticos. Nenhum modelo em uso foi substituído. Pesos padrão e caches completos ficam separados no Drive. A confiança permanece em 0,25; não é uma nova busca de limiares. Não há formulário para preencher.\n',encoding='utf-8')
    return summary,selection
