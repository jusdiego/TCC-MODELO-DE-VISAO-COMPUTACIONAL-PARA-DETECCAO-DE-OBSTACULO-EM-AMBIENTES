"""Relatório final, contagens e texto-base sem seleção adaptativa de modelos."""
from __future__ import annotations
import html,json
from pathlib import Path
import diag11_core as core
from final20_data import MODELS,digest

def pct(v):return 'indefinido (sem denominador)' if v is None else f'{100*v:.2f}%'.replace('.',',')
def pp(v):return 'indefinido' if v is None else f'{100*v:+.2f} p.p.'.replace('.',',')
def ci(row,delta=False):
    if row['ic95_inferior'] is None:return 'IC indefinido'
    format_value=pp if delta else pct
    return f"[{format_value(row['ic95_inferior'])}; {format_value(row['ic95_superior'])}]"
def table(rows):
    if not rows:return '<p>Sem registros.</p>'
    keys=list(rows[0])
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(k))+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table></div>'

def build(records,evaluations,intervals,boot,groups,extraction,plan,expected,context,output):
    output=Path(output);find=lambda model,metric:next(r for r in intervals if r['escopo']=='global' and r['modelo_ou_comparacao']==model and r['metrica']==metric)
    rows=[];shown=[];classes=[]
    for model in MODELS:
        result=evaluations[model];s=result['summary'];a=s['alertas_por_objeto']
        if s['imagens']!=plan['images'] or not result['cache_manifest']['concluido']:raise ValueError('Resultado final incompleto.')
        row={'modelo':model,'imagens':s['imagens'],**a,'alertas_GT':a['tp']+a['fn'],'alertas_pred_avaliaveis':a['tp']+a['fp'],
            'omissoes_altas':s['alta_sem_alerta'],'GT_prioridade_alta':s['alta_ref'],'taxa_omissoes_altas':s['taxa_alta_sem_alerta'],
            'GT_sem_cobertura':s['gt_ignorados_sem_cobertura'],'previsoes_sem_cobertura_no_limiar':s['predicoes_sem_cobertura_no_limiar']}
        for metric in ['precision','recall','f1','taxa_alta_sem_alerta']:
            interval=find(model,metric)
            for k in ['ic95_inferior','ic95_superior','replicas_validas']:row[metric+'_'+k]=interval[k]
        rows.append(row)
        shown.append({'Modelo':model,'Precisão':pct(a['precision']),'Recall':pct(a['recall']),'F1':pct(a['f1']),
            'TP / FN / FP':f"{a['tp']} / {a['fn']} / {a['fp']}",
            'Omissões altas / GT alto':f"{s['alta_sem_alerta']} / {s['alta_ref']}",'Taxa omissões altas':pct(s['taxa_alta_sem_alerta'])})
        classes += [{'modelo':model,**r} for r in result['classes']]
    if rows[0]['alertas_GT']!=rows[1]['alertas_GT'] or rows[0]['GT_prioridade_alta']!=rows[1]['GT_prioridade_alta']:
        raise ValueError('Modelos avaliados com denominadores GT diferentes.')
    core.save_csv(output/'comparacao_final_20.csv',rows)
    core.save_csv(output/'resultados_por_classe_20.csv',classes)
    delta=find('delta_candidato10_menos_baseline','recall');primary=find('baseline','recall')
    primary_rows=[r for r in intervals if r['estimando_primario']]
    core.save_csv(output/'estimandos_primarios_20.csv',primary_rows)
    limitations=list(plan['limits'])+[
        'Cobertura parcial: previsões de classes sem anotação não são julgadas como acertos ou erros.',
        'A reserva é uma amostra planejada para diversidade de classes; não estima automaticamente a frequência de obstáculos no uso cotidiano.',
        'As validações Home e externa de desenvolvimento já influenciaram os ajustes e são apresentadas separadamente.',
        'Classes com menos de 30 GT são sinalizadas; métricas sem denominador são indefinidas.',
        'A política V2 usa classe, posição e tamanho aparente; as métricas de alerta medem concordância com essa regra.'
    ]
    if extraction['pares_semelhanca_phash_com_desenvolvimento']:
        limitations.append('Há semelhanças pHash com imagens de desenvolvimento registradas em extracao_reserva_20.json; interpretar a separação com essa ressalva. Nenhuma imagem foi substituída.')
    summary={'version':'20-avaliacao-final-1.0.0','concluido':True,'status':'avaliacao_final_concluida',
        'fase_desenvolvimento_encerrada':True,'context_key':context,'pacote_fonte_19_sha256':expected['pacote_19'],
        'protocolo_sha256':digest(plan),'dataset_sha256':plan['dataset_sha256'],'modelos':list(MODELS),
        'pesos_sha256':plan['weight_sha256'],'modelo_principal_predefinido':'baseline','comparador_predefinido':'candidato_10',
        'reserva_imagens':len(records),'reserva_avaliada':True,'reserva_inferida':True,
        'treinamento_executado':False,'pesos_ou_limiares_ajustados':False,'modelo_substituido':False,
        'test_home_inferido':False,'test_oi20k_inferido':False,'preenchimento_manual_obrigatorio':False,
        'estimandos_primarios':primary_rows,'bootstrap':boot,'grupos':groups['grupos'],
        'resultados':rows,'limites':limitations,'decisao':'Encerrar a fase experimental e relatar metodologia, resultados e limitações; sem nova seleção de pesos na reserva.'}
    core.save_json(output/'resumo_avaliacao_final_20.json',summary)
    text='# Avaliação final dos modelos e encerramento experimental\n\n'
    text+=f"A avaliação final utilizou os {len(records)} IDs previamente reservados do conjunto externo, com {groups['grupos']} grupos de imagens para a análise de incerteza. Foram avaliados o baseline, definido como principal, e o candidato 10, definido como comparador, segundo o protocolo fixado no notebook 19 antes da consulta à reserva. Não houve treinamento ou ajuste de pesos e limiares nesta etapa.\n\n"
    text+='A política V2, a confiança de 0,25 e a associação por classe com IoU mínimo de 0,5 foram mantidas. A captura usou Ultralytics 8.4.150, confiança 0,001, entrada 640, NMS IoU 0,70, limite de 300 propostas, batch 1 e FP32. As métricas de alertas medem concordância com a política aplicada às caixas anotadas nas classes com cobertura.\n\n'
    text+='| Modelo | Precisão | Recall | F1 | TP | FN | FP | Omissões altas / GT alto |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for r in rows:
        text+=f"| {r['modelo']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} | {r['tp']} | {r['fn']} | {r['fp']} | {r['omissoes_altas']} / {r['GT_prioridade_alta']} |\n"
    text+=f"\nO recall do baseline foi {pct(primary['estimativa'])}, com IC de 95% {ci(primary)}. A diferença pareada de recall, candidato 10 menos baseline, foi {pp(delta['estimativa'])}, com IC de 95% {ci(delta,True)}. Esse intervalo descreve a incerteza da diferença neste protocolo, sem estabelecer superioridade universal ou autorizar substituição automática do modelo.\n\n"
    text+=f"Os intervalos utilizaram 2.000 reamostragens pareadas de grupos, com reposição e semente {plan['confidence_intervals']['seed']}. Imagens com bytes iguais ou pHash64 com distância de Hamming até 4 integram o mesmo componente transitivo. Cada réplica utilizou os mesmos grupos para ambos os modelos. As métricas foram calculadas como razões das contagens agregadas, mantendo réplicas sem denominador como indefinidas. No recall principal do baseline e na diferença de recall, houve {primary['replicas_validas']} e {delta['replicas_validas']} réplicas válidas, respectivamente.\n\n"
    text+='Os resultados por classe, os suportes e os intervalos completos estão nas tabelas anexas. As 12 classes permanecem documentadas, inclusive aquelas que não geram alertas pela V2. As métricas de detecção no limiar de 0,25 também são fornecidas separadamente das métricas de alerta. Os valores absolutos de omissões da validação de 1.000 imagens não foram usados como limites de aprovação para a reserva de 300.\n\n'
    text+='A independência se refere à reserva preservada no fluxo documentado. O histórico completo do treinamento original do baseline não está comprovado e não permite garantir ausência de exposição em todo pré-treino. A amostra também não representa automaticamente a distribuição de cenas do uso cotidiano. A avaliação não mede distância real ou segurança de locomoção.\n\n'
    text+='A fase experimental foi encerrada com a caracterização dos dois modelos predefinidos. Os resultados de desenvolvimento e os resultados finais devem ser apresentados em tabelas separadas. A continuação do trabalho consiste em redigir a metodologia, discutir os resultados e limitações e utilizar a demonstração do notebook 19 como material ilustrativo.\n'
    (output/'texto_base_resultados_finais_20.md').write_text(text,encoding='utf-8')
    main_intervals=[]
    for model in [*MODELS,'delta_candidato10_menos_baseline']:
        for metric in ['precision','recall','f1','taxa_alta_sem_alerta']:
            row=find(model,metric);isdelta=model.startswith('delta_')
            main_intervals.append({'Modelo / comparação':model,'Métrica':metric,'Estimativa':pp(row['estimativa']) if isdelta else pct(row['estimativa']),
                'IC 95%':ci(row,isdelta),'Réplicas válidas':f"{row['replicas_validas']}/2000",'Primário':'sim' if row['estimando_primario'] else 'secundário'})
    class_shown=[{'Modelo':r['modelo'],'Classe':r['classe'],'GT':r['gt_avaliaveis'],'Imagens com cobertura':r['imagens_com_cobertura'],
        'Suporte <30 GT':'sim' if r['suporte_reduzido'] else 'não','Precisão alertas':pct(r['precision']),'Recall alertas':pct(r['recall']),
        'Precisão detector':pct(r['detector_precision']),'Recall detector':pct(r['detector_recall'])} for r in classes]
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Avaliação final 20</title><style>body{font:16px/1.6 system-ui;background:#eef2f6;color:#162739;margin:0}main{max-width:1450px;margin:auto;padding:28px}.note{padding:20px;background:white;border:1px solid #cbd5e1;border-radius:12px;margin:20px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:white}th,td{padding:10px;border:1px solid #d1dae4;text-align:left}h1,h2{line-height:1.25}a{color:#1d4ed8}</style><main><p>TCC · Encerramento experimental</p><h1>Avaliação final na reserva externa</h1>'''
    page+=f'<div class="note"><strong>{len(records)} imagens · {groups["grupos"]} grupos · dois modelos predefinidos</strong><p>Protocolo fixado no notebook 19. Confiança 0,25, política V2, IoU de associação 0,5. Nenhum treinamento ou ajuste após a consulta à reserva.</p><p>Fase experimental concluída. Os dois modelos foram caracterizados sem seleção automática para uso.</p></div>'
    page+='<h2>Resultados finais dos alertas</h2>'+table(shown)
    page+='<h2>Incerteza e comparação pareada</h2><p>Bootstrap de 2.000 réplicas com os mesmos grupos para ambos os modelos; intervalos percentis de 95%. Diferenças em pontos percentuais. Denominadores nulos permanecem indefinidos.</p>'+table(main_intervals)
    page+='<h2>Resultados por classe e cobertura</h2><p>As métricas consideram somente classes com referência na imagem. Classes sem cobertura permanecem sem julgamento. Os intervalos por classe estão no CSV completo. Resultados secundários são descritivos.</p>'+table(class_shown)
    page+='<h2>Limitações</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in limitations)+'</ul>'
    page+='<h2>Arquivos para o relato</h2><ul>'+''.join(f'<li><a href="{n}">{n}</a></li>' for n in ['comparacao_final_20.csv','estimandos_primarios_20.csv','resultados_por_classe_20.csv','intervalos_confianca_20.csv','texto_base_resultados_finais_20.md','fontes19/comparacao_finais_e_referencias.csv','fontes19/protocolo_avaliacao_final.json'])+'</ul></main></html>'
    (output/'relatorio_avaliacao_final_20.html').write_text(page,encoding='utf-8')
    (output/'LEIA_RESULTADO.md').write_text('# Avaliação final 20 concluída\n\nAbra relatorio_avaliacao_final_20.html para os resultados. O texto-base está em texto_base_resultados_finais_20.md. As tabelas principais são comparacao_final_20.csv, resultados_por_classe_20.csv e intervalos_confianca_20.csv.\n\nEnvie pacote_avaliacao_final_20.zip para a análise e a escrita do TCC. Não ajuste os modelos ou limiares usando esta reserva. Nenhum novo treinamento é necessário para encerrar esta fase.\n',encoding='utf-8')
    return summary
