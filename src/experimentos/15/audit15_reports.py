"""Relatórios descritivos da auditoria, preservando a origem dos sinais."""
from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import audit15_core as core
import audit15_models as models


def html_table(rows):
    if not rows:return '<p>Sem registros neste recorte.</p>'
    fields=list(rows[0]);out='<div class="scroll"><table><thead><tr>'
    out+=''.join('<th>'+html.escape(str(k))+'</th>' for k in fields)+'</tr></thead><tbody>'
    for row in rows:out+='<tr>'+''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>' for k in fields)+'</tr>'
    return out+'</tbody></table></div>'


def font(size):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def render_case(record,semantic,path):
    with Image.open(record['image']) as f: image=f.convert('RGB')
    draw=ImageDraw.Draw(image);size=max(12,min(image.size)//45);ft=font(size)
    evidence={r['index']:r for r in semantic['objects'] if r['kind']=='gt'}
    strong=[]
    for g in record['gt']:
        sem=evidence.get(g['index'],{});bad=sem.get('status')=='divergencia_forte_ia'
        color='#f45175' if bad else '#43c9a0'
        box=tuple(max(0,min(record['width'] if i%2==0 else record['height'],x)) for i,x in enumerate(g['bbox']))
        draw.rectangle(box,outline=color,width=max(2,size//6))
        label=f"{g['line']}: {g['classe']}"
        if bad:label+=' | IA: '+sem.get('top_tight','?');strong.append(g)
        x,y=box[:2]
        extent=draw.textbbox((0,0),label,font=ft,stroke_width=1)
        x=min(x,max(0,image.width-(extent[2]-extent[0])-3))
        draw.text((x,max(0,y-size-3)),label,fill=color,font=ft,stroke_width=1,stroke_fill='#111827')
    canvas=Image.new('RGB',(1280,650),'#0c1727')
    overview=ImageOps.contain(image,(630,580));canvas.paste(overview,((640-overview.width)//2,50+(580-overview.height)//2))
    focus=strong[0] if strong else next((g for g in record['gt'] if g['classe_id'] in core.FOCUS),None)
    if focus:
        with Image.open(record['image']) as f: crop=models.crop_square(f.convert('RGB'),focus['bbox'],1.2)
        crop=ImageOps.contain(crop,(580,580));canvas.paste(crop,(660+(600-crop.width)//2,50+(580-crop.height)//2))
    header=ImageDraw.Draw(canvas)
    header.text((15,12),'Rótulos originais: verde. Divergência forte da IA: rosa.',fill='white',font=font(18))
    header.text((660,12),'Recorte com contexto; sinal automático, sem correção.',fill='white',font=font(17))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=86)


def report(output,records,semantics,predictions,duplicates,metadata):
    output=Path(output); image_rows=[];box_rows=[];extra_rows=[];scene_counts=Counter()
    duplicate_ids={r[k] for r in duplicates for k in ['id','similar_id']}
    for r in records:
        sem=semantics.get(r['id']); signals=[i['code'] for i in r['issues']]
        by_gt={o['index']:o for o in sem['objects'] if o['kind']=='gt'} if sem else {}
        det={o['gt_index']:o for o in sem['detector']['gt']} if sem else {}
        strong=0;uncertain=0;focus_difficult=0
        for g in r['gt']:
            obj=by_gt.get(g['index'],{});status=obj.get('status','nao_analisado_por_bloqueio')
            strong+=status=='divergencia_forte_ia';uncertain+=status in {'inconclusivo_ia','recorte_pequeno_sem_analise','recorte_invalido_sem_analise'}
            if g['classe_id'] in core.FOCUS and det.get(g['index'],{}).get('matched_at_025') is False:focus_difficult+=1
            row={'id':r['id'],'source':r['source'],'line':g['line'],'classe_id':g['classe_id'],'classe':g['classe'],
                 'bbox_original':json.dumps(g['bbox']),'area_relativa':g['area_relativa'],'menor_lado_em_640':g['menor_lado_em_640'],
                 'cobertura_classe':bool(r['known_mask'] & (1 << g['classe_id'])),
                 'sinais_geometria':'|'.join(g['flags']),'triagem_ia':status,
                 'top_ia_recorte':obj.get('top_tight',''),'top_ia_contexto':obj.get('top_context',''),
                 'margem_vs_rotulo_recorte':obj.get('margin_over_label_tight'),
                 'margem_vs_rotulo_contexto':obj.get('margin_over_label_context'),
                 'correspondencia_B_025':det.get(g['index'],{}).get('matched_at_025'),
                 'confianca_B_correspondente':det.get(g['index'],{}).get('matched_score')}
            box_rows.append(row)
        if r['id'] in duplicate_ids:signals.append('duplicacao_ou_semelhanca_em_train')
        if strong:signals.append('divergencia_semantica_forte_ia')
        if focus_difficult:signals.append('objeto_foco_sem_correspondencia_do_B')
        if sem:
            sc=sem['scene'];scene_counts[(r['source'],sc['top'],sc['indicative'])]+=1
            if sc['indicative'] and sc['top']!='interior':signals.append('contexto_visual_possivelmente_fora_de_interior')
            for i,p in enumerate(sem['detector']['extras']):
                obj=next((o for o in sem['objects'] if o['kind']=='extra' and o['bbox']==p['bbox'] and o['classe_id']==p['classe_id']),{})
                possible_missing=bool(p['known_class'] and p['best_iou_any_label']<.1 and obj.get('status')=='compativel_ia')
                extra_rows.append({'id':r['id'],'source':r['source'],'classe':core.NAMES[p['classe_id']],
                    'bbox_pred':json.dumps(p['bbox']),'confidence':p['confianca'],'cobertura_classe':p['known_class'],
                    'melhor_iou_qualquer_rotulo':p['best_iou_any_label'],'triagem_recorte_ia':obj.get('status','nao_selecionado_para_clip'),
                    'possivel_objeto_sem_caixa_na_classe_coberta':possible_missing,'interpretation':p['interpretation']})
        status=('bloqueado_ou_quarentena' if r['status_input']!='legivel' else
                'com_divergencia_semantica_forte' if strong else 'com_incerteza_semantica' if uncertain else 'sem_divergencia_forte_identificada')
        image_rows.append({'id':r['id'],'source':r['source'],'lote':r['lote'],'sha256':r['sha256'],
            'known_mask':r['known_mask'],'status_entrada':r['status_input'],'caixas_validas':len(r['gt']),
            'caixas_com_divergencia_forte_ia':strong,'caixas_inconclusivas_ou_pequenas':uncertain,
            'objetos_foco_sem_correspondencia_do_B':focus_difficult,'triagem_imagem':status,
            'sinais':'|'.join(sorted(set(signals))),'contexto_ia':sem['scene']['top'] if sem else '',
            'contexto_ia_indicativo':sem['scene']['indicative'] if sem else None,
            'revisao_humana_realizada':False,'rotulo_corrigido':False})
    core.save_csv(output/'auditoria_imagens.csv',image_rows)
    core.save_csv(output/'auditoria_caixas.csv',box_rows)
    core.save_csv(output/'previsoes_sem_associacao.csv',extra_rows)
    core.save_csv(output/'duplicacoes_treino.csv',duplicates,['id','similar_id','type','distance'])
    issues=[{'id':r['id'],'source':r['source'],**i} for r in records for i in r['issues']]
    fields=list(dict.fromkeys(['id','source','line','code','blocking']+[k for row in issues for k in row]))
    core.save_csv(output/'sinais_estruturais.csv',issues,fields)
    class_rows=[]
    for source in sorted({r['source'] for r in records}):
        for cid,name in enumerate(core.NAMES):
            subset=[b for b in box_rows if b['source']==source and b['classe_id']==cid]
            counts=Counter(b['triagem_ia'] for b in subset)
            class_rows.append({'source':source,'classe':name,'caixas_validas':len(subset),
                               'compativeis_ia':counts['compativel_ia'],'divergencias_fortes_ia':counts['divergencia_forte_ia'],
                               'inconclusivas_ia':counts['inconclusivo_ia'],
                               'pequenas_ou_invalidas_sem_analise':counts['recorte_pequeno_sem_analise']+counts['recorte_invalido_sem_analise'],
                               'sem_analise_por_bloqueio':counts['nao_analisado_por_bloqueio'],
                               'sem_correspondencia_B':sum(b['correspondencia_B_025'] is False for b in subset)})
    core.save_csv(output/'resumo_por_classe.csv',class_rows)
    wardrobes=Counter((b['source'],b['top_ia_recorte'],b['triagem_ia']) for b in box_rows if b['classe_id']==7)
    wardrobe_rows=[{'source':s,'conceito_mais_proximo':c or 'sem_analise','triagem_ia':st,'caixas':n}
                   for (s,c,st),n in sorted(wardrobes.items())]
    core.save_csv(output/'wardrobe_hipoteses_semanticas.csv',wardrobe_rows,['source','conceito_mais_proximo','triagem_ia','caixas'])
    domains=[{'source':s,'contexto_mais_proximo_ia':c,'margem_indicativa':flag,'imagens':n} for (s,c,flag),n in sorted(scene_counts.items())]
    core.save_csv(output/'contextos_visuais_ia.csv',domains)
    source_counts=Counter(r['source'] for r in records)
    summary={'versao':core.VERSION,'concluido':True,'status':'auditoria_concluida_sem_alteracao_de_dados',
             'imagens_catalogadas':len(records),'imagens_por_fonte':dict(source_counts),
             'imagens_analisadas_pelos_modelos':len(semantics),'status_entrada':dict(Counter(r['status_input'] for r in records)),
             'caixas_validas':len(box_rows),'triagem_caixas':dict(Counter(r['triagem_ia'] for r in box_rows)),
             'problemas_estruturais':dict(Counter(r['code'] for r in issues)),
             'pares_duplicacao_ou_semelhanca_registrados':len(duplicates),
             'previsoes_sem_associacao_classe_coberta':sum(r['cobertura_classe'] for r in extra_rows),
             'previsoes_sem_associacao_classe_desconhecida':sum(not r['cobertura_classe'] for r in extra_rows),
             'possiveis_objetos_sem_caixa_triagem_ia':sum(r['possivel_objeto_sem_caixa_na_classe_coberta'] for r in extra_rows),
             'novos_treinamentos':0,'rotulos_alterados':False,'modelo_substituido':False,
             'novo_dataset_aprovado':False,'preenchimento_manual_obrigatorio':False,'revisao_humana_realizada':False,
             'test_home_extraido':False,'oi_test_extraido':False,'reserva_futura_extraida':False,
             'validacoes_inferidas':False,'reserva_futura_inferida':False,
             'proveniencia_treino_original_baseline':'nao_determinado',
             'limites':['Contagens são sinais automáticos; não estimam erros reais dos rótulos.',
                       'Recorte Open Images dos cinco lotes do piloto; não representativo dos 80 mil.',
                       'YOLO viu parte destes dados no treino; sua concordância não valida generalização.',
                       'CLIP congelado pode errar; margens heurísticas não são probabilidades calibradas.',
                       'pHash de 64 bits com distância até 4 registra até cinco vizinhos por imagem; é um sinal de semelhança e pode gerar falsos positivos.',
                       'Home assume cobertura completa conforme protocolo anterior; a completude real não foi comprovada.',
                       'pHash protegido externo usa apenas metadados já preservados; Home VAL/TEST e OI TEST têm proteção por SHA, sem nova revisão perceptual.',
                       'Sem avaliação independente de risco físico ou utilidade para locomoção.'],
             'execucao':metadata}
    core.save_json(output/'resumo_auditoria_15.json',summary)
    # Galeria dirigida de até 48 imagens, estratificada por fonte e sinal.
    lookup={r['id']:r for r in records}; candidates=[r for r in image_rows if r['id'] in semantics]
    pools=defaultdict(list)
    for row in candidates:
        cat=('semantica' if row['caixas_com_divergencia_forte_ia'] else
             'dificuldade_detector' if row['objetos_foco_sem_correspondencia_do_B'] else
             'geometria_contexto' if row['sinais'] else 'comparacao_sem_sinal')
        pools[(row['source'],cat)].append(row)
    for key,rows in pools.items():rows.sort(key=lambda r:core.digest(r['id']))
    selected=[]
    for i in range(48):
        keys=sorted(pools)
        if not keys:break
        key=keys[i%len(keys)];selected.append((key,pools[key].pop(0)))
        if not pools[key]:del pools[key]
    cards=[];gallery=[]
    for index,((source,category),row) in enumerate(selected):
        relative=f'galeria/caso_{index:03d}.jpg';render_case(lookup[row['id']],semantics[row['id']],output/relative)
        gallery.append({'id':row['id'],'source':source,'categoria':category,'arquivo':relative})
        cards.append('<article><h3>'+html.escape(source+' · '+category)+'</h3><p class="id">'+html.escape(row['id'])+'</p><img loading="lazy" src="'+relative+'" alt="Caixas originais e recorte com contexto"><p>'+html.escape(row['sinais'] or 'Sem sinal automático nesta seleção.')+'</p></article>')
    core.save_json(output/'indice_galeria.json',gallery)
    brief=f"Auditoria concluída: {len(records):,} imagens catalogadas; {len(semantics):,} analisadas pelos modelos; {len(box_rows):,} caixas válidas.".replace(',','.')
    focus_rows=[r for r in class_rows if r['classe'] in {core.NAMES[c] for c in core.FOCUS}]
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Auditoria 15 — Dados de treino</title><style>body{font:16px/1.55 system-ui;background:#eef3f8;color:#16263a;margin:0}main{max-width:1400px;margin:auto;padding:30px}h1{line-height:1.15}article,.note{background:white;padding:20px;border:1px solid #d9e3ed;border-radius:12px;margin:20px 0}img{width:100%;height:auto}table{border-collapse:collapse;background:white;width:100%}th,td{border:1px solid #d9e3ed;padding:10px;text-align:left}.scroll{overflow:auto}.id{font-size:12px;overflow-wrap:anywhere}a{color:#1163aa}</style><main><p>TCC · Visão computacional</p><h1>Auditoria automática dos dados de treino</h1>'''
    page+='<div class="note"><strong>'+brief+'</strong><p>Nenhum rótulo, modelo ou conjunto de avaliação foi alterado. Os sinais abaixo orientam o próximo experimento; não são uma taxa de erro confirmada das anotações.</p></div>'
    page+='<h2>Classes prioritárias</h2>'+html_table(focus_rows)
    page+='<p>Todas as 12 classes foram auditadas nas imagens elegíveis. A divergência forte exige o mesmo conceito alternativo nos dois recortes e margem mínima congelada; concordância com CLIP também não comprova o rótulo.</p>'
    page+='<h2>Hipóteses para wardrobe</h2><p>A hipótese estreita é um móvel alto para roupas. Armários genéricos, estantes, racks e cômodas são conceitos alternativos de triagem, sem alteração da taxonomia original.</p>'+html_table(wardrobe_rows)
    page+='<h2>Contexto visual — hipótese por IA</h2>'+html_table(domains)
    page+='<h2>Limites</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary['limites'])+'</ul>'
    page+='<h2>Próximo passo</h2><p>Enviar o pacote desta auditoria para decidir um recorte ou uma correção documentada antes de outro treino. Não há formulário para preencher. Uma seleção futura pode preservar rótulos e máscaras e remover imagens inteiras; sinais CLIP não serão tratados como novas anotações verdadeiras.</p>'
    page+='<h2>Exemplos selecionados automaticamente</h2><p>A galeria é dirigida por sinais e não estima sua frequência na base.</p>'+''.join(cards)+'</main></html>'
    (output/'relatorio_auditoria_15.html').write_text(page,encoding='utf-8')
    (output/'LEIA_RESULTADO.md').write_text('# Resultado do notebook 15\n\n'+brief+'\n\nEnvie pacote_auditoria_15.zip. Não é necessário preencher fichas ou repetir os testes.\n\nAbra relatorio_auditoria_15.html após extrair o ZIP. As contagens descrevem sinais por IA e verificações estruturais, não erros de rótulo confirmados. Os originais permanecem preservados; não há novo dataset aprovado ou treinamento.\n',encoding='utf-8')
    return summary
