"""Relatórios automáticos e galeria do diagnóstico de confiança."""
from __future__ import annotations
import html
import json
from pathlib import Path
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
import diag11_core as core
import diag11_reports as r11
import conf14_evaluation as evaluation
import conf14_data as data

LABELS={'home_val_historica':'Home VAL histórica','openimages_val_nova':'Open Images VAL de desenvolvimento'}

def pct(value):return '—' if value is None else f'{100*value:.2f}%'

def comparison_rows(results,bundle,selection):
    rows=[]
    for cohort in data.COHORTS:
        anchor=results[cohort][.25]['summary'];a=anchor['alertas_por_objeto']
        for model in ['baseline','candidato_10']:
            s=bundle['evaluations'][model][cohort]['summary'];m=s['alertas_por_objeto']
            rows.append({'coorte':cohort,'modelo':model,'conf':.25,'origem':'registro autenticado do notebook 12',
                'imagens':s['imagens'],**m,'alta_ref':s['alta_ref'],'alta_sem_alerta':s['alta_sem_alerta'],
                'delta_fp_vs_B_025':m['fp']-a['fp'],'delta_fn_vs_B_025':m['fn']-a['fn'],
                'elegivel_14':False})
        for entry in selection['assessed']:
            conf=entry['conf'];s=results[cohort][conf]['summary'];m=s['alertas_por_objeto']
            rows.append({'coorte':cohort,'modelo':'B_e03','conf':conf,'origem':'cache da nova inferência do notebook 14',
                'imagens':s['imagens'],**m,'alta_ref':s['alta_ref'],'alta_sem_alerta':s['alta_sem_alerta'],
                'delta_fp_vs_B_025':m['fp']-a['fp'],'delta_fn_vs_B_025':m['fn']-a['fn'],
                'elegivel_14':entry['elegivel']})
    return rows

def plot_curves(output,results,bundle):
    fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True,layout='constrained')
    for col,cohort in enumerate(data.COHORTS):
        for row,key in enumerate(['precision','recall','fp','alta_sem_alerta']):
            values=[]
            for conf in evaluation.THRESHOLDS:
                s=results[cohort][conf]['summary']
                v=s[key] if key=='alta_sem_alerta' else s['alertas_por_objeto'][key]
                values.append(100*v if row<2 and v is not None else v)
            ax=axes[row,col];ax.plot(evaluation.THRESHOLDS,values,'o-',color='#176ca4',label='B_e03')
            for model,color in [('baseline','#6c7785'),('candidato_10','#d17920')]:
                s=bundle['evaluations'][model][cohort]['summary']
                v=s[key] if key=='alta_sem_alerta' else s['alertas_por_objeto'][key]
                if v is not None:ax.axhline(100*v if row<2 else v,color=color,linestyle='--',label=model+' / 0,25')
            ax.axvline(.25,color='#97a1ad',linewidth=1,alpha=.6)
            ax.grid(alpha=.2)
            ax.set_ylabel(['Precisão de alertas (%)','Recall de alertas (%)','Falsos alertas avaliáveis','Objetos de prioridade alta sem alerta'][row])
            if row<2:ax.set_ylim(0,100)
            else:
                ax.set_ylim(bottom=0);ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0,col].set_title(LABELS[cohort]);axes[0,col].legend(fontsize=8)
        axes[3,col].set_xticks(evaluation.THRESHOLDS)
        axes[3,col].set_xlabel('Limiar global de confiança')
    fig.suptitle('Pesos B_e03 fixos · política V2 fixa · uma inferência por imagem',fontsize=13)
    fig.savefig(Path(output)/'curvas_confianca.png',dpi=150)
    plt.close(fig)

def gallery(output,cohorts,caches,results,focus,policy,seed=42):
    output=Path(output);index=[];cards=[]
    for cohort,records in cohorts.items():
        chosen=r11.select_gallery_cases(results[cohort][.25]['images'],results[cohort][focus]['images'],limit=12,seed=seed)
        mapping={r['id']:r for r in records}
        for i,item in enumerate(chosen):
            r=mapping[item['id']];pred=caches[cohort][r['id']]
            gt=core.with_policy([g for g in r['gt'] if r['known_mask']&(1<<g['classe_id'])],r['width'],r['height'],policy)
            original=core.with_policy([p for p in pred if p['confianca']>=.25],r['width'],r['height'],policy)
            changed=core.with_policy([p for p in pred if p['confianca']>=focus],r['width'],r['height'],policy)
            panels=[r11.draw_panel(r,gt,'Referência V2 avaliável'),
                    r11.draw_panel(r,original,'B_e03 / confiança 0,25'),
                    r11.draw_panel(r,changed,f'B_e03 / confiança {focus:.2f}')]
            canvas=Image.new('RGB',(1620,470),'#101a28')
            for j,panel in enumerate(panels):canvas.paste(panel,(j*540,0))
            name=f'galeria/{cohort}_{i:02}.jpg';path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
            canvas.save(path,quality=86)
            index.append({**item,'coorte':cohort,'conf_comparada':focus,'arquivo':name,
                          'creditos':r.get('creditos',{}),'preenchimento_manual':False})
            cards.append('<article><h3>'+html.escape(LABELS[cohort]+' · '+item['categoria'])+'</h3><img loading="lazy" src="'+name+'" alt="Referência e previsões do mesmo modelo em dois limiares"><p>'+html.escape(r['name'])+'</p></article>')
    core.save_json(output/'indice_galeria.json',index)
    return ''.join(cards)

def selection_table(selection):
    head=['Confiança','P Home','R Home','Alta omitida Home','P externa','R externa','Alta omitida externa','Cumpre todos os requisitos']
    table='<table><thead><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr></thead><tbody>'
    for entry in selection['assessed']:
        h,e=entry['home'],entry['external'];ha,ea=h['alertas_por_objeto'],e['alertas_por_objeto']
        values=[f"{entry['conf']:.2f}",pct(ha['precision']),pct(ha['recall']),h['alta_sem_alerta'],pct(ea['precision']),pct(ea['recall']),e['alta_sem_alerta'],'Sim, nas validações' if entry['elegivel'] else 'Não']
        table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    return table+'</tbody></table>'

def build_report(output,results,bundle,selection,reproduction,cohorts,caches,policy,config):
    output=Path(output)
    rows=comparison_rows(results,bundle,selection)
    core.save_csv(output/'comparacao_confianca.csv',rows)
    core.save_json(output/'selecao_limiar.json',selection)
    checks=[{'conf':e['conf'],**e['checks'],'comparavel_historico':e['comparavel_historico'],
             'elegivel_final':e['elegivel']} for e in selection['assessed']]
    core.save_csv(output/'requisitos_por_limiar.csv',checks)
    changed=[];losses=[];priority=[]
    for cohort in data.COHORTS:
        a,b,c=evaluation.effects(results,bundle,cohort,caches[cohort]);changed+=a;losses+=b;priority+=c
    core.save_csv(output/'alertas_que_mudaram_vs_B_025.csv',changed,
        ['coorte','conf','id','classe','bbox_ref','alerta_B_025','alerta_no_limiar','prioridade_ref','confianca_associada'])
    core.save_csv(output/'perdas_vs_candidato10_por_limiar.csv',losses,
        ['coorte','conf','id','classe','bbox_ref','prioridade_ref','recuperado','evento_B_025','evento_no_limiar','melhor_iou_cache_mesma_classe','confianca_dessa_proposta','limite'])
    core.save_csv(output/'perdas_com_objeto_ainda_detectado.csv',priority,
        ['coorte','id','classe','bbox_ref','prioridade_candidato10','prioridade_B_025','confianca_candidato10','confianca_B_025','observacao'])
    recovery=[]
    for cohort in data.COHORTS:
        for conf in evaluation.THRESHOLDS:
            part=[r for r in losses if r['coorte']==cohort and r['conf']==conf]
            high=[r for r in part if r['prioridade_ref']=='alto']
            recovery.append({'coorte':cohort,'conf':conf,'perdas_B_025_vs_candidato10':len(part),
                'dessas_perdas_recuperadas':sum(r['recuperado'] for r in part),
                'perdas_altas_B_025_vs_candidato10':len(high),
                'dessas_perdas_altas_recuperadas':sum(r['recuperado'] for r in high)})
    core.save_csv(output/'recuperacao_perdas_resumo.csv',recovery)
    plot_curves(output,results,bundle)
    cards=gallery(output,cohorts,caches,results,selection['limiar_galeria'],policy)
    if not selection['comparabilidade_confirmada']:
        status='As contagens/associações em 0,25 divergiram do experimento 12. Os resultados são diagnósticos e nenhum limiar é sugerido.'
    elif selection['limiar_sugerido'] is None:
        status='Nenhum limiar da grade cumpriu todos os requisitos. O baseline continua em uso.'
    else:
        status=f"O limiar {selection['limiar_sugerido']:.2f} cumpriu os requisitos nas validações de desenvolvimento. É uma configuração candidata, sem substituição automática do baseline."
    p=config['protocolo']['regra_original_12'];h=p['home'];e=p['externa']
    refs='<table><tr><th>Validação</th><th>Referência em 0,25</th><th>Precisão</th><th>Recall</th><th>Alta sem alerta</th></tr>'
    for cohort in data.COHORTS:
        for model in ['baseline','candidato_10']:
            s=bundle['evaluations'][model][cohort]['summary'];a=s['alertas_por_objeto']
            refs+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [LABELS[cohort],model,pct(a['precision']),pct(a['recall']),s['alta_sem_alerta']])+'</tr>'
    refs+='</table>'
    reproduction_table='<table><tr><th>Validação</th><th>Divergências por imagem</th><th>Divergências por objeto</th><th>Reprodução confirmada</th></tr>'
    for r in reproduction:
        reproduction_table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [LABELS[r['coorte']],r['divergencias_imagem'],r['divergencias_objetos'],'Sim' if r['reproduziu'] else 'Não'])+'</tr>'
    reproduction_table+='</table>'
    css='body{font:16px/1.6 system-ui;background:#f3f6fa;color:#182638;margin:0}main{max-width:1320px;margin:auto;padding:28px}h1{line-height:1.2}h2{margin-top:30px}article,.note{background:white;border:1px solid #d6e0eb;border-radius:10px;padding:18px;margin:18px 0}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{border:1px solid #d6e0eb;padding:9px;text-align:left}.scroll{overflow:auto}img{width:100%;height:auto}a{color:#12689c}code{overflow-wrap:anywhere}'
    page='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TCC — Diagnóstico 14</title><style>'+css+'</style><main><h1>14 — Diagnóstico automático de confiança</h1>'
    page+='<div class="note"><strong>'+html.escape(status)+'</strong><p>Pesos B_e03 e política V2 fixos; zero novos treinos. Sem formulários. As referências do baseline e do candidato 10 são os registros autenticados do notebook 12, sem nova inferência desses dois modelos.</p></div>'
    page+='<h2>Grade fixa e requisitos preservados</h2><p>Um único limiar global deve funcionar nas duas validações. Home exige precisão superior a '+pct(h['precision_maior_que'])+', recall de pelo menos '+pct(h['recall_minimo'])+' e no máximo '+str(h['alta_sem_alerta_maximo'])+' omissões altas. A externa exige precisão de pelo menos '+pct(e['precision_minima'])+', recall de pelo menos '+pct(e['recall_minimo'])+' e no máximo '+str(e['alta_sem_alerta_maximo'])+' omissões altas. O JSON conserva os valores exatos.</p>'
    page+='<div class="scroll">'+selection_table(selection)+'</div><h2>Troca entre recuperação e falsos alertas</h2><img src="curvas_confianca.png" alt="Precisão, recall, falsos alertas e omissões altas em função do limiar">'
    page+='<p><a href="comparacao_confianca.csv">Comparação completa</a> · <a href="recuperacao_perdas_resumo.csv">Recuperação das perdas anteriores</a> · <a href="requisitos_por_limiar.csv">Requisitos por limiar</a></p>'
    page+='<p>As propostas numéricas do modelo foram preservadas, inclusive caixas de área zero. Os arquivos caches/&lt;coorte&gt;/diagnostico_geometria_previsoes.json registram geometrias incomuns. Não houve recorte ou remoção automática de propostas.</p>'
    page+='<h2>Reprodução em 0,25</h2><div class="scroll">'+reproduction_table+'</div><p>A confirmação exige igualdade dos contadores por imagem e dos estados por objeto. Qualquer divergência impede sugerir limiar por esses requisitos históricos; as diferenças ficam nos CSVs de reprodução.</p>'
    page+='<h2>Referências autenticadas do experimento 12</h2><div class="scroll">'+refs+'</div>'
    page+='<h2>Casos selecionados automaticamente</h2><p>Referência, B em 0,25 e B no limiar '+f"{selection['limiar_galeria']:.2f}"+'. Sem limiar elegível, o painel usa 0,10, definido previamente apenas para diagnóstico. A galeria dirigida não estima prevalência de erros e não pede preenchimento.</p>'+cards
    page+='<h2>Limites</h2><p>Esta é seleção de ponto de operação em validações já usadas no desenvolvimento, não teste final independente nem prova de calibração probabilística. A origem histórica do treino do baseline permanece não determinada. Open Images tem cobertura parcial por imagem/classe; somente classes avaliáveis entram nas métricas. Confiança menor pode aumentar falsos alertas e alterar associações. As previsões são pós-NMS com max_det=300. A política V2 nas caixas não constitui validação independente de risco físico. As 300 imagens reservadas e os testes ficaram sem extração/inferência nesta etapa. Os rótulos não foram modificados; a auditoria anterior por IA não é uma aprovação humana da base.</p></main></html>'
    (output/'relatorio_diagnostico_14.html').write_text(page,encoding='utf-8')
    summary={'versao':evaluation.VERSION,'concluido':True,'status':selection['status'],'mensagem':status,
        'limiar_sugerido':selection['limiar_sugerido'],'grade':list(evaluation.THRESHOLDS),
        'reproducao_confirmada':selection['comparabilidade_confirmada'],
        'modelo_inferido':'B_e03','imagens_avaliadas':sum(len(v) for v in cohorts.values()),
        'imagens_inferidas_nesta_execucao':config.get('imagens_inferidas_nesta_execucao'),
        'novos_treinamentos':0,'preenchimento_manual_obrigatorio':False,
        'baseline_e_candidato10_reinferidos':False,'referencias':'pacote 12 autenticado',
        'modelo_em_uso_continua_baseline':True,'modelos_substituidos':False,
        'reserva_futura_extraida':False,'reserva_futura_inferida':False,
        'test_home_inferido':False,'test_oi20k_inferido':False,'rotulos_alterados':False,
        'validacoes_usadas_na_selecao':True,'nao_e_teste_final_independente':True,
        'correcao_cache':'14-cache-validacao-1.0.1',
        'geometria_previsoes':{c:m.get('geometria_resumo',{}) for c,m in config.get('caches',{}).items()},
        'peso_B_e03_sha256':config['peso_B_e03_sha256'],'dataset_sha256':config['dataset_sha256'],
        'recuperacao_perdas':recovery,
        'proximo_passo':('Congelar a configuração candidata e preparar avaliação futura independente, sem uso automático da reserva.' if selection['limiar_sugerido'] is not None else
                        'Analisar as diferenças de reprodução antes de selecionar configuração.' if not selection['comparabilidade_confirmada'] else
                        'Encerrar esta hipótese de limiar; planejar revisão automática mais ampla dos dados e novo experimento documentado.')}
    core.save_json(output/'resumo_diagnostico_14.json',summary)
    (output/'LEIA_RESULTADO.md').write_text('# Diagnóstico 14\n\n'+status+'\n\nAbra relatorio_diagnostico_14.html. As comparações e a galeria já foram geradas; não há fichas para preencher. Envie pacote_diagnostico_14.zip pelo chat.\n\nO pacote inclui as previsões completas pós-NMS em confiança 0,001, para permitir novas análises sem repetir a inferência. Nenhum treinamento foi executado ou modelo em uso foi substituído.\n',encoding='utf-8')
    return summary

def package_reports(output):
    output=Path(output);archive=output/'pacote_diagnostico_14.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path!=archive and not path.name.endswith('.tmp') and path.suffix not in {'.pt','.pkl'}:
                z.write(path,path.relative_to(output))
    return archive
