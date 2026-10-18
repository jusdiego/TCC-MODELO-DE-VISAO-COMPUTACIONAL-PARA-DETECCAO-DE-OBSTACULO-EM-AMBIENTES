"""Resultados de desenvolvimento; meta simultânea sem promessa de teste final."""
from pathlib import Path
import html,json,shutil
import diag11_core as core
from learn21_protocol import SPEC,digest

def pct(value):return 'indefinido' if value is None else f'{100*value:.2f}%'.replace('.',',')
def table(rows):
    if not rows:return '<p>Sem registros.</p>'
    cols=list(rows[0]);return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c,'')))+'</td>' for c in cols)+'</tr>' for r in rows)+'</tbody></table></div>'

def build(output,run,selected,summaries,profiles,context,expected,spec=SPEC):
    output=Path(output);prep=json.loads((output/'preparacao_dados_21.json').read_text())
    best=json.loads((output/'melhor_modelo_21.json').read_text())
    source=output/best['weight_relative']
    if core.sha256_file(source)!=best['weight_sha256']:raise ValueError('Peso escolhido mudou antes do relatório.')
    shutil.copy2(source,output/'melhor_modelo.pt')
    ready=bool(run['concluido']);target=bool(selected['target_met'])
    status='treino_pausado_retomavel' if not ready else 'treino_concluido_meta_atingida_validacao' if target else 'treino_concluido_meta_nao_atingida'
    rows=[];comparison=[];classes=[]
    for cohort,s in summaries.items():
        row={'modelo':'yolov8s_21','coorte':cohort,'confianca':selected['threshold'],'imgsz':spec['imgsz'],
            **s['detector'],'gt_avaliaveis':s['detector']['tp']+s['detector']['fn'],
            **{'alertas_'+k:v for k,v in s['alertas'].items()},'alta_ref':s['alta_ref'],'alta_sem_alerta':s['alta_sem_alerta'],
            'meta_80_70':s['target_met']}
        rows.append(row)
        classes += [{'coorte':cohort,'modelo':'yolov8s_21','confianca':selected['threshold'],**c} for c in s['classes']]
        prior=output/'referencia19'/(cohort+'_candidato10.json')
        if prior.is_file():
            r=json.loads(prior.read_text());d=r['detector_no_limiar']
            comparison.append({'modelo':'candidato_10','coorte':cohort,'confianca':.25,'imgsz':640,
                'precision':d['precision'],'recall':d['recall'],'f1':d['f1']})
        comparison.append({k:row[k] for k in ['modelo','coorte','confianca','imgsz','precision','recall','f1']})
    grid=[{'confianca':r['threshold'],**r['detector'],'fitness':r['fitness'],'meta_80_70':r['target_met']} for r in profiles]
    core.save_csv(output/'comparacao_desenvolvimento_21.csv',comparison)
    core.save_csv(output/'resultados_21.csv',rows);core.save_csv(output/'classes_21.csv',classes);core.save_csv(output/'grade_limiares_21.csv',grid)
    summary={'version':spec['version'],'concluido':ready,'status':status,'context_key':context,'training':run,
        'meta_precision':spec['target_precision'],'meta_recall':spec['target_recall'],'meta_atingida_validacao_externa':target,
        'meta_atingida_teste_final':None,'teste_final_novo_consultado':False,'reserva20_usada_treino_ou_selecao':False,
        'modelo':'yolov8s','pretrained_url':expected['pretrained_url'],'pretrained_sha256':expected['pretrained_sha256'],
        'melhor_epoch_desenvolvimento':best['epoch'],'peso_final_sha256':best['weight_sha256'],'confianca_escolhida':selected['threshold'],
        'resultados':rows,'treino_home':prep['train_home'],'treino_oi':prep['train_openimages'],
        'selecao':'Peso escolhido pelas validações expostas; limiar recalculado no peso exportado, ainda em desenvolvimento.',
        'modelo_substituido':False,'preenchimento_manual_obrigatorio':False,
        'limites':['Atingir 80/70 na validação não comprova o resultado em imagens novas.',
            'Rótulos parciais: somente classes conhecidas entram nas métricas; não é precisão de todas as mensagens.',
            'Mudanças simultâneas em capacidade, resolução, amostragem e orçamento; não isolam a causa de eventual ganho.',
            'O sistema não foi validado para distância, perigo físico ou segurança de locomoção.']}
    core.save_json(output/'resumo_treinamento_21.json',summary)
    shown=[{'Modelo':r['modelo'],'Conjunto':r['coorte'],'Confiança':r['confianca'],'Entrada':r['imgsz'],
        'Precisão detector':pct(r['precision']),'Recall detector':pct(r['recall']),'F1 detector':pct(r['f1'])} for r in comparison]
    class_shown=[{'Conjunto':r['coorte'],'Classe':r['classe'],'GT':r['gt_avaliaveis'],'P':pct(r['precision']),'R':pct(r['recall']),
        'TP / FN / FP':f"{r['tp']} / {r['fn']} / {r['fp']}",'Suporte <30':'sim' if r['suporte_reduzido'] else 'não'} for r in classes]
    message='Treino pausado por limite de tempo. Execute Run All novamente para retomar.' if not ready else 'Meta 80% de precisão e 70% de recall atingida na validação externa. Falta nova avaliação final.' if target else 'Treino concluído. A meta conjunta 80/70 não foi atingida na validação externa.'
    text=f'''# Rodada 21 — YOLOv8s e supervisão ampliada

{message}

Foram usados {prep['train_home']} exemplos Home e {prep['train_openimages']} Open Images, selecionados do TRAIN histórico. As 12 classes e as anotações humanas foram preservadas. Os quatro arquivos Home da quarentena histórica permaneceram excluídos. Foram aplicadas exclusões por coincidência com conjuntos protegidos e semelhança perceptual, documentadas na preparação.

O modelo partiu de pesos públicos do YOLOv8s. O plano prevê até {spec['epochs']} épocas, AdamW com taxa inicial {spec['lr0']}, redução cossenoidal, entrada {spec['imgsz']}, batch {spec['batch']} e cobertura parcial da classificação. Misturas de imagens foram desativadas para preservar essa cobertura. A amostragem destina {spec['home_per_batch']} posições a Home e {spec['oi_per_batch']} a Open Images por batch; em OI, combina passagem uniforme e repetição por classe. Uma época é um orçamento amostrado e não garante exposição a todos os IDs.

Foram concluídas {run['epochs']} épocas. O peso selecionado no desenvolvimento veio da época {best['epoch']}. O limiar no peso exportado foi {selected['threshold']}. O critério combina o atendimento simultâneo de precisão 80% e recall 70% na validação externa, com desempate por F1 e menor limiar. Não foram descartadas classes para atingir a meta.

'''
    text+='| Modelo | Validação | Confiança | Precisão detector | Recall detector | F1 detector |\n| --- | --- | ---: | ---: | ---: | ---: |\n'
    for r in comparison:text+=f"| {r['modelo']} | {r['coorte']} | {r['confianca']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |\n"
    text+=f'\nAs validações contêm {prep["val_home"]} imagens Home e {prep["val_external"]} externas já expostas ao desenvolvimento. As 300 imagens da avaliação 20 não foram usadas para treino ou seleção nesta rodada. O resultado permanece de desenvolvimento; outra reserva será necessária para confirmar o desempenho final da nova versão. As métricas de detecção e alertas estão separadas.\n'
    (output/'texto_resultados_desenvolvimento_21.md').write_text(text,encoding='utf-8')
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Treinamento 21</title><style>body{font:16px/1.6 system-ui;margin:0;background:#eef2f6;color:#162739}main{max-width:1450px;margin:auto;padding:28px}.note{padding:20px;background:white;border:1px solid #cbd5e1;border-radius:10px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;border:1px solid #cbd5e1;text-align:left}h1,h2{line-height:1.25}</style><main><p>TCC · Nova linha de desenvolvimento</p><h1>YOLOv8s: precisão e recuperação de objetos</h1>'''
    page+='<div class="note"><strong>'+html.escape(message)+'</strong><p>'+f"{prep['train_home']} Home + {prep['train_openimages']} Open Images. Épocas concluídas: {run['epochs']}. Melhor época: {best['epoch']}."+'</p></div>'
    page+='<h2>Comparação nas validações de desenvolvimento</h2><p>O candidato 10 usou entrada 640 e confiança 0,25. A versão 21 usa entrada 768 e limiar escolhido na validação; a comparação caracteriza as configurações completas.</p>'+table(shown)
    page+='<h2>Detecção por classe</h2>'+table(class_shown)+'<h2>Limitações</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in summary['limites'])+'</ul>'
    page+='<p>Envie pacote_treinamento_21.zip. Os checkpoints de retomada permanecem no Drive; o pacote inclui um peso de inferência e os resultados.</p></main></html>'
    (output/'relatorio_treinamento_21.html').write_text(page,encoding='utf-8')
    return summary
