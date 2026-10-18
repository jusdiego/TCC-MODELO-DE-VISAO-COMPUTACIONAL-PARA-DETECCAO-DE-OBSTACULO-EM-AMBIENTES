"""Tabelas consolidadas, texto para o TCC e pacote da demonstração."""
from __future__ import annotations
import html,json
from pathlib import Path
import diag11_core as core
from exp12_protocol import judge
import cons19_protocol as protocol

def table(rows):
    if not rows:return '<p>Sem registros.</p>'
    keys=list(rows[0]);return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(k))+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table></div>'
def percentage(v):return 'sem suporte' if v is None else f'{100*v:.2f}%'.replace('.',',')

def consolidate(bundle,output):
    output=Path(output);rows=[]
    entries=[]
    for model in protocol.MODELS:
        refs=bundle['references'][model]
        entries.append({'experiment':'referencia','id':model,'epoch':0,'updates':0,
            'home':refs[protocol.COHORTS[0]]['summary'],'external':refs[protocol.COHORTS[1]]['summary'],'reference':True})
    entries += [{**e,'reference':False} for e in bundle['entries']]
    for e in entries:
        checks=None if e['reference'] else judge(e['home'],e['external'],bundle['rule'])
        row={'experimento':e['experiment'],'modelo':e['id'],'epoca':e['epoch'],'atualizacoes':e['updates'],
            'papel':'referencia fixa' if e['reference'] else 'final predefinido' if e['epoch']==3 else 'diagnostico intermediario'}
        for side,prefix in [('home','home'),('external','externa')]:
            s=e[side];row.update({prefix+'_'+k:v for k,v in s['alertas_por_objeto'].items()})
            row[prefix+'_alta_sem_alerta']=s['alta_sem_alerta'];row[prefix+'_alta_ref']=s['alta_ref']
        row['criterios_atendidos']=None if checks is None else sum(v for k,v in checks.items() if k not in {'elegivel','retencao_externa'})
        row['elegivel_pelos_seis']=None if checks is None else checks['elegivel']
        rows.append(row)
    core.save_csv(output/'comparacao_todos_checkpoints.csv',rows)
    finals=[r for r in rows if r['papel']!='diagnostico intermediario']
    core.save_csv(output/'comparacao_finais_e_referencias.csv',finals)
    experiments=[{'experimento':16,'hipotese':'Wardrobe desconhecida em Home durante o ajuste',
        'desfecho':'Nenhum final elegível; efeito pequeno entre os dois braços'},
        {'experimento':17,'hipotese':'Limiares por classe para cadeira e mesa','desfecho':'0/49 perfis elegíveis na calibração; nenhuma nova escolha na verificação'},
        {'experimento':18,'hipotese':'Aumentar OI por batch de 4 para 8','desfecho':'Pequeno ganho Home; retenção externa não preservada; 2/6 critérios no final 8/8'}]
    core.save_csv(output/'resumo_hipoteses_16_18.csv',experiments)
    return rows,finals,experiments

def finish(bundle,evaluations,checks,selection,final_plan,context,output):
    output=Path(output);rows,finals,experiments=consolidate(bundle,output)
    shown=[]
    for r in finals:
        shown.append({'Modelo':r['modelo'],'P Home':percentage(r['home_precision']),'R Home':percentage(r['home_recall']),
            'Altas omitidas Home':r['home_alta_sem_alerta'],'P externa':percentage(r['externa_precision']),
            'R externa':percentage(r['externa_recall']),'Altas omitidas externas':r['externa_alta_sem_alerta']})
    gt_counts={c:len(v) for c,v in bundle['cohorts'].items()}
    summary={'version':protocol.VERSION,'concluido':True,'status':'consolidacao_e_demonstracao_concluidas',
        'context_key':context,'modelo_principal':'baseline','comparador':'candidato_10','novos_pesos':False,
        'treinamento_executado':False,'inferencia_executada':False,'previsoes_reutilizadas':True,
        'validacoes':gt_counts,'imagens_demonstracao':len(selection),'galeria_ilustrativa':True,
        'selecao_demonstracao_sha256':protocol.digest(selection),'protocolo_final_sha256':protocol.digest(final_plan),
        'reserva_futura_extraida':False,'reserva_futura_inferida':False,'test_home_inferido':False,'test_oi20k_inferido':False,
        'modelo_substituido':False,'preenchimento_manual_obrigatorio':False,'referencias_reproduzidas':all(r['reproduziu'] for r in checks),
        'decisao':'Encerrar os ajustes curtos desta linha; consolidar resultados e demonstrar o comportamento dos dois modelos fixos.',
        'limites':['As validações já influenciaram o desenvolvimento; não são teste final independente.',
            'A galeria não mede desempenho e inclui exemplos de acertos e falhas por regras fixas.',
            'Cobertura parcial OI: classe desconhecida não vira falso positivo nem negativo conhecido.',
            'V2 usa geometria aparente e classe, sem medir distância ou risco físico validado.',
            'Origem do treinamento original do baseline permanece não determinada.']}
    core.save_json(output/'resumo_consolidacao_19.json',summary)
    text='''# Consolidação dos experimentos e demonstração do sistema

O desenvolvimento avaliou um detector inicial, denominado baseline, e o candidato 10, ajustado com imagens Home e Open Images. A avaliação de alertas usou a política V2, confiança de 0,25 e associação por classe com IoU mínimo de 0,5. As métricas de alerta representam concordância com essa política, sem validação independente de risco físico.

Foram mantidas duas validações de desenvolvimento: 456 imagens Home e 1.000 imagens externas. A avaliação externa respeitou a cobertura parcial das anotações: classes sem cobertura não foram presumidas como negativas. Esses conjuntos já influenciaram o desenvolvimento e não constituem teste final independente.

'''
    text+='| Modelo | Precisão Home | Recall Home | Omissões altas Home | Precisão externa | Recall externo | Omissões altas externas |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for r in finals:text+='| '+r['modelo']+' | '+' | '.join([percentage(r['home_precision']),percentage(r['home_recall']),str(r['home_alta_sem_alerta']),percentage(r['externa_precision']),percentage(r['externa_recall']),str(r['externa_alta_sem_alerta'])])+' |\n'
    text+='''
No experimento 16, o ajuste da supervisão de armários em Home apresentou efeito pequeno e nenhum checkpoint final cumpriu os seis requisitos históricos. No 17, nenhuma das 49 combinações de limiares de cadeira e mesa foi elegível na calibração; por esse motivo, nenhuma nova configuração foi submetida à verificação interna. No 18, o batch equilibrado 8/8 apresentou pequeno ganho em Home sobre 12/4, sem preservar o recall e as omissões altas na validação externa. Aumentar a exposição a Open Images nesse orçamento curto não produziu a retenção externa pretendida.

O baseline foi mantido como referência principal, e o candidato 10 como comparador relevante para o domínio externo. Essa decisão não afirma superioridade universal do baseline. Os demais modelos continuam documentados, sem escolher retrospectivamente uma época intermediária para substituir os desfechos finais.

Foi preparada uma demonstração com as mesmas imagens para referência anotada, baseline e candidato 10. A seleção usa categorias fixas de acertos, omissões, falsos alertas e diferenças entre os modelos, com ordenação determinística por hash. A galeria é ilustrativa; as métricas da tabela usam todas as imagens das duas validações. As mensagens consideram classe, posição e tamanho aparente segundo a V2; não medem distância real.

A reserva externa de 300 imagens permanece sem extração ou inferência nesta consolidação. O protocolo de avaliação posterior fixa os dois pesos, a política e os limiares antes de sua consulta. Prevê métricas e denominadores completos, comparação pareada e intervalos de confiança por grupos de imagens. Os limites absolutos de omissões da validação de 1.000 imagens não serão transportados para a reserva de 300. A consulta futura não autoriza ajustes retrospectivos para melhorar o mesmo resultado.

As limitações incluem uma semente e orçamento curto nos ajustes, diferenças de domínio e de anotação, cobertura parcial externa e histórico original do treino do baseline não comprovado. Os experimentos não demonstram que o conjunto completo de 100 mil imagens seja inadequado, pois o ajuste 18 usou somente o replay de 2 mil imagens Open Images. A demonstração é acadêmica e controlada, sem comprovação de segurança assistiva.
'''
    (output/'texto_base_capitulo4.md').write_text(text,encoding='utf-8')
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Consolidação 19 — resultados e demonstração</title><style>body{font:16px/1.6 system-ui;background:#eef2f6;color:#162739;margin:0}main{max-width:1450px;margin:auto;padding:28px}.note{background:white;border:1px solid #d3dce5;border-radius:12px;padding:20px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border:1px solid #d3dce5;text-align:left}h1,h2{line-height:1.2}a{color:#1d4ed8}</style><main><p>TCC · Consolidação experimental</p><h1>Resultados, demonstração e avaliação posterior</h1><div class="note"><strong>Baseline como referência principal; candidato 10 como comparador.</strong><p>Os ajustes 16–18 não produziram um novo modelo elegível nos seis critérios congelados. Esta etapa reaproveita previsões e não treina ou consulta testes.</p><p><a href="demonstracao_19.html">Abrir a demonstração comparativa</a> · <a href="texto_base_capitulo4.md">Texto-base para o capítulo experimental</a></p></div><h2>Finais e referências nas duas validações</h2>'''
    page+=table(shown)+'<h2>Hipóteses encerradas nesta linha</h2>'+table(experiments)
    page+='<h2>Avaliação final planejada</h2><p>300 imagens externas do papel reserva_futura, sem consulta nesta etapa. Pesos, limiares, política, associação e análise estatística foram fixados em protocolo_avaliacao_final.json. Baseline e candidato 10 serão avaliados, sem escolha adaptativa ou troca automática de modelo.</p>'
    page+='<h2>Limites</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in summary['limites'])+'</ul></main></html>'
    (output/'relatorio_consolidacao_19.html').write_text(page,encoding='utf-8')
    (output/'LEIA_RESULTADO.md').write_text('# Notebook 19 concluído\n\nAbra demonstracao_19.html para comparar os dois modelos nas imagens. Abra relatorio_consolidacao_19.html para as tabelas. O texto-base está em texto_base_capitulo4.md. A avaliação final está planejada em protocolo_avaliacao_final.json, sem abrir a reserva.\n\nEnvie pacote_consolidacao_19.zip. Nenhum novo treino, inferência ou ajuste manual foi executado.\n',encoding='utf-8')
    return summary
