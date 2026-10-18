"""Relato separado: seis classes, máscaras visíveis e reserva própria desta linha."""
from pathlib import Path
import html
import nyu22_common as c

def pct(v):return 'indefinido' if v is None else f'{100*v:.2f}%'.replace('.',',')
def table(rows):
    if not rows:return '<p>Ainda não avaliado.</p>'
    keys=list(rows[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table></div>'
def build(job,run,best,val,test,spec=c.TRAIN_SPEC):
    output=Path(job['output']);prep=job['info']['report']
    status='pausado_retomavel' if not run['completed'] else 'concluido_meta_atingida_teste' if test and test['target_met'] else 'concluido_meta_nao_atingida_teste'
    summary={'version':spec['version'],'completed':bool(run['completed'] and test is not None),'status':status,'run':run,
        'context':job['context'],'dataset_id':job['info']['dataset_id'],'classes':c.NAMES,'class_names_pt':c.PT_NAMES,'old_project_class_ids':c.GLOBAL_IDS,
        'model':'yolov8s','best_epoch':best['epoch'],'weight_sha256':best['weight_sha256'],'threshold':val['threshold'],
        'goal':{'precision':spec['target_precision'],'recall':spec['target_recall']},'validation':val,'test':test,
        'target_met_validation':val['target_met'],'target_met_test':test['target_met'] if test else None,
        'test_used_for_selection':False,'same_benchmark_as_21':False,'existing_model_replaced':False,'manual_labels_required':False,
        'train_images':prep['train_images'],'val_images':prep['val_images'],'test_images':prep['test_images_reserved'],
        'limits':prep['limits']+['Atingir a meta nesta base e em seis classes não garante o mesmo resultado nas 12 classes nem em fotos do usuário.',
            'Treino usa somente RGB; as caixas vêm de máscaras humanas de partes visíveis. Não são caixas amodais de objetos completos.',
            'Não valida distância nem segurança de locomoção.']}
    c.save(output/'resumo_nyu23.json',summary)
    comparison=[];classes=[]
    for label,s in [('Validação',val),('Teste reservado',test)]:
        if s is None:continue
        comparison.append({'Conjunto':label,'Imagens':s['images'],'Precisão':pct(s['detector']['precision']),'Recall':pct(s['detector']['recall']),'F1':pct(s['detector']['f1']),'Meta 80/70':'sim' if s['target_met'] else 'não'})
        for row in s['classes']:classes.append({'Conjunto':label,'Classe':c.PT_NAMES[row['class_id']],'GT':row['gt'],'Precisão':pct(row['precision']),'Recall':pct(row['recall']),'F1':pct(row['f1']),'TP / FN / FP':f'{row["tp"]} / {row["fn"]} / {row["fp"]}'})
    c.csv_save(output/'comparacao_val_teste.csv',comparison);c.csv_save(output/'classes_val_teste.csv',classes)
    message='Treino pausado; execute Run All novamente para retomar. A reserva não foi avaliada.' if not run['completed'] else 'Treino e teste concluídos. Meta conjunta atingida no teste desta linha.' if test and test['target_met'] else 'Treino e teste concluídos. Meta conjunta não atingida no teste desta linha.'
    text=f'''# Rodada compacta NYU — seis classes

{message}

O detector YOLOv8s foi treinado a partir de pesos públicos COCO, em RGB, para cama, sofá, cadeira, mesa, porta e janela. A base contém 1.449 imagens; nesta preparação ficaram {prep['train_images']} em TRAIN, {prep['val_images']} em VAL e {prep['test_images_reserved']} no TEST oficial. TRAIN/VAL foram separados em grupos de cenas e semelhança perceptual dentro do TRAIN oficial. A escolha da divisão usou representatividade de rótulos, antes de treinar, sem resultados de modelos ou rótulos de TEST.

As caixas foram obtidas automaticamente da extensão visível de cada par de classe original e instância nas máscaras humanas NYU, incluindo instância zero quando existe, como no toolbox original. Não houve anotação manual nem criação de rótulos por previsão. Imagens sem objetos das seis classes foram mantidas. Pixels sem rótulo existem na origem.

Foram concluídas {run['epochs']} épocas; a melhor selecionada foi {best['epoch']}. A confiança escolhida na validação foi {val['threshold']}. O critério tenta atender precisão 80% e recall 70% simultaneamente, com IoU 0,5 e as seis classes mantidas. No teste, peso e confiança permaneceram fixos. O teste só acontece após o fim do treino; uma execução pausada não abre suas previsões.

'''
    text+='| Conjunto | Precisão | Recall | F1 |\n| --- | ---: | ---: | ---: |\n'
    for r in comparison:text+=f'| {r["Conjunto"]} | {r["Precisão"]} | {r["Recall"]} | {r["F1"]} |\n'
    text+='\nOs percentuais desta rodada não são comparáveis diretamente aos da rodada 21: mudaram base, classes e definição das caixas. SUN RGB-D incorpora imagens NYU; esta linha começa de pesos públicos e não usa pesos Home/C10/21. A reserva serve para a avaliação desta nova linha, não para afirmar independência de versões anteriores treinadas em Home/SUN.\n'
    if test:text+=f'\nAP padrão Ultralytics: mAP50={pct(test["native_ap"]["map50"])}; mAP50–95={pct(test["native_ap"]["map50_95"])}. AP usa uma curva de confiança e não equivale à precisão no limiar operacional. Intervalos de 95% por reamostragem de cenas constam no JSON; a indicação de meta usa a estimativa pontual.\n'
    (output/'texto_resultados_nyu23.md').write_text(text,encoding='utf-8')
    page='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYU compacto</title><style>body{font:16px/1.6 system-ui;margin:0;background:#eef2f6;color:#173046}main{max-width:1300px;margin:auto;padding:28px}.note{background:white;padding:20px;border:1px solid #cbd5e1;border-radius:12px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}th,td{padding:9px;border:1px solid #cbd5e1;text-align:left}</style><main><h1>NYU: detector de seis classes</h1>'
    page+='<div class="note"><strong>'+html.escape(message)+'</strong><p>'+f'Épocas: {run["epochs"]}. Melhor época: {best["epoch"]}. Confiança: {val["threshold"]}.'+'</p></div>'
    page+='<h2>Validação e teste</h2>'+table(comparison)+'<h2>Resultados por classe</h2>'+table(classes)
    page+='<h2>Limitações</h2><ul>'+''.join('<li>'+html.escape(t)+'</li>' for t in summary['limits'])+'</ul><p>Envie pacote_treino_nyu23.zip. Não houve substituição automática do modelo anterior.</p></main></html>'
    (output/'relatorio_nyu23.html').write_text(page,encoding='utf-8');return summary
