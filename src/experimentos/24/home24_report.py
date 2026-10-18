"""Relatório legível e pacote de resultados do 24."""
from pathlib import Path
import html, json
import home24_common as c

def pct(v):return f'{100*v:.2f}%'.replace('.',',') if v is not None else 'indefinido'
def build(job,result):
    out=job['output'];rows=[]
    for split,key in [('Home VAL','validation'),('Home TEST histórico','test_historico')]:
        for name,profile in result[key].items():
            d=profile['detector'];a=profile['alertas'];rows.append({'conjunto':split,'modelo':name,'imagens':profile['images'],'confianca':profile['threshold'],
              **d,'alertas_precision':a['precision'],'alertas_recall':a['recall'],'alertas_f1':a['f1'],
              'alta_ref':profile['alta_ref'],'alta_sem_alerta':profile['alta_sem_alerta'],'meta_80_70':profile['target_met']})
    c.csv_save(out/'comparacao_24.csv',rows)
    classes=[]
    for name,p in result['test_historico'].items():classes += [{'modelo':name,**r} for r in p['classes']]
    c.csv_save(out/'classes_test_historico_24.csv',classes)
    t=result['test_historico']['rfdetr_small24'];s=result['test_historico_secondary']['rfdetr_small24'];run=result['run'];best=result['best']
    status='Meta atingida no TEST Home histórico' if t['target_met'] else 'Meta principal não atingida no TEST Home histórico'
    text=f'''# Experimento 24 — RF-DETR Small no Home

**{status}.**

Foram usados {len(job['train'])} exemplos TRAIN, {len(job['val'])} VAL e {len(job['test_descriptors'])} TEST Home histórico. As 12 classes foram mantidas. Os rótulos humanos foram reutilizados, com exclusões históricas e proteção contra coincidências documentadas no manifesto. Nenhuma rotulagem manual foi exigida.

O treinamento partiu do checkpoint público COCO do RF-DETR Small, entrada {job['spec']['resolution']}, EMA e precisão mista FP16 na GPU. Foram registradas {run['epochs']} épocas; encerramento: {run['reason']}. O checkpoint escolhido por mAP50–95 de VAL veio da época {best['epoch']}; sua confiança de operação, escolhida pelo fitness micro de VAL, foi {t['threshold']}.

| Conjunto | Modelo | Confiança | Precisão | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
'''
    for r in rows:text+=f"| {r['conjunto']} | {r['modelo']} | {r['confianca']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |\n"
    text+=f'''
Nas 12 classes do TEST histórico, RF-DETR obteve P={pct(t['detector']['precision'])}, R={pct(t['detector']['recall'])} e F1={pct(t['detector']['f1'])}. No desfecho secundário das seis classes da V2, no mesmo limiar, obteve P={pct(s['detector']['precision'])}, R={pct(s['detector']['recall'])} e F1={pct(s['detector']['f1'])}. O desfecho secundário não substitui o principal.

Os alertas V2 obtiveram P={pct(t['alertas']['precision'])}, R={pct(t['alertas']['recall'])} e F1={pct(t['alertas']['f1'])}; {t['alta_sem_alerta']} de {t['alta_ref']} objetos com prioridade alta de referência ficaram sem alerta. Essas métricas medem concordância com a regra heurística nas caixas, sem validar distância ou segurança de locomoção.

O TEST Home já havia sido consultado em fases anteriores do projeto. Nesta rodada, os pesos e limiares foram congelados antes da avaliação; isso preserva a separação desta execução, mas não transforma o conjunto em uma reserva nova. Os intervalos usam bootstrap pareado por imagem, sem agrupamento validado de cenas; correlação entre fotos pode estreitá-los. Comparações entre RF-DETR e YOLO descrevem configurações completas, incluindo resoluções e pós-processamentos diferentes, e não isolam somente a arquitetura.

As fontes permanecem as mesmas do projeto. O Open Images não participou do novo treinamento. A cobertura declarada para Home não equivale a garantia de perfeição das anotações. Resultados restritos ao Home não confirmam generalização para uma câmera em movimento.
'''
    ext=result.get('external_nyu',{})
    if ext.get('executed'):
        d=ext['metrics']['detector'];text+=f"\nO diagnóstico NYU avaliou {ext['images']} imagens em seis classes comuns, após exclusões de coincidência com Home: P={pct(d['precision'])}, R={pct(d['recall'])}, F1={pct(d['f1'])}. Peso e limiar foram mantidos. A diferença de anotação e os recortes impedem tratar essa linha como comparação direta com o 23 ou prova de independência entre fontes.\n"
    else:text+='\nDiagnóstico externo NYU: '+str(ext.get('reason','não executado'))+'\n'
    text+='''
## Arquivos principais

- `modelo_rfdetr24.pth`: peso RF-DETR de inferência com configuração e nomes das classes; não é arquivo YOLO `.pt`.
- `decisao_antes_test_24.json`: pesos, confianças e referência congelados antes desta avaliação de TEST.
- `resumo_experimento_24.json`: resultados completos e estado do experimento.
- `comparacao_24.csv`, `classes_test_historico_24.csv`: comparação e resultados por classe.
- `bootstrap_test_historico_24.json`: intervalos e diferenças pareadas, incluindo alertas.
- `prova_tempo_24.json`: tempos medidos de épocas, validação e persistência.
- `retomada/slot_0` e `retomada/slot_1`: estados completos no Drive; não entram no ZIP de análise.

Para análise, enviar `pacote_experimento_24.zip` completo.
'''
    (out/'relatorio_24.md').write_text(text,encoding='utf-8')
    cells=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]) if k not in {'precision','recall','f1'} else pct(r[k]))+'</td>' for k in ['conjunto','modelo','confianca','precision','recall','f1'])+'</tr>' for r in rows)
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Experimento 24</title><style>body{font:16px/1.6 system-ui;background:#eef2f6;color:#162739;margin:0}main{max-width:1150px;margin:auto;padding:30px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:10px;text-align:left}.scroll{overflow:auto}</style><main><h1>RF-DETR Small · Home</h1><section><h2>'''+html.escape(status)+'''</h2><p>TEST histórico já exposto. Seleção desta rodada somente em VAL. Desfecho principal: 12 classes; secundário: seis classes V2.</p></section><section class="scroll"><table><tr><th>Conjunto</th><th>Modelo</th><th>Confiança</th><th>Precisão</th><th>Recall</th><th>F1</th></tr>'''+cells+'''</table></section><section><p>O relatório Markdown e o resumo JSON contêm resultados por classe, alertas, bootstrap, histórico de treino e limites de interpretação.</p></section></main></html>'''
    (out/'relatorio_24.html').write_text(page,encoding='utf-8')
    print(status,flush=True);print('RF-DETR TEST histórico:',{k:pct(t['detector'][k]) for k in ['precision','recall','f1']},flush=True)
    return out/'relatorio_24.html'
