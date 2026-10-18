"""Relatório do controle 12 e galeria de rótulos sem alteração automática."""
from __future__ import annotations
import html
from pathlib import Path
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import diag11_core as core
import diag11_reports as r11
from exp12_protocol import order_id


def semantic_gallery(home, replay, output, policy, limit=12, seed=42):
    output = Path(output)
    rows, cards = [], []
    for source, records in [('Home TRAIN',[r for r in home if r['role']=='home_train']),
                            ('Open Images TRAIN',list(replay.values()))]:
        candidates = [r for r in records if any(o['classe_id']==7 for o in r['gt'])]
        candidates.sort(key=lambda r:order_id(r['id'],seed))
        for i, record in enumerate(candidates[:limit]):
            if record['role'] not in {'home_train','train'}:
                raise RuntimeError('A galeria de rótulos deve usar somente treino.')
            relative = f'galeria_rotulos/{source.split()[0]}_{i:02}.jpg'
            path = output/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            objects = core.with_policy(record['gt'],record['width'],record['height'],policy)
            r11.draw_panel(record,objects,source+' / wardrobe').save(path,quality=90)
            rows.append({'id':record['id'],'fonte':source,'imagem':record['name'],
                         'arquivo':relative,'anotacao_parece_correta':'nao_revisado',
                         'mesmo_conceito_de_guarda_roupa':'nao_revisado',
                         'fotografia_real':'nao_revisado','observacao':''})
            cards.append('<article><h3>'+html.escape(source)+'</h3><img src="'+relative+'" alt="Rótulos originais do treino"><p>'+html.escape(record['name'])+'</p></article>')
    core.save_csv(output/'revisao_rotulos_wardrobe.csv',rows)
    page = '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Revisão de rótulos</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#f5f6f8}article{display:inline-block;vertical-align:top;margin:8px;background:white;padding:12px}img{width:500px;max-width:100%}</style><h1>Guarda-roupa: revisão dos dados de treino</h1><p>As caixas são as originais. Armários de parede, ilustrações e diferenças de significado entre bases precisam de revisão humana. Nenhum rótulo foi alterado neste experimento. Esta amostra por hash não estima a frequência de erros.</p>'+''.join(cards)+'</html>'
    (output/'revisao_rotulos.html').write_text(page,encoding='utf-8')
    return rows


def checkpoint_rows(selection):
    rows=[]
    for entry in selection['assessed']:
        row={k:entry[k] for k in ['id','arm','epoch','updates','weight_sha256']}
        for key,prefix in [('home','home'),('external','externa')]:
            s=entry[key];a=s['alertas_por_objeto']
            row.update({prefix+'_'+k:a[k] for k in ['tp','fn','fp','precision','recall','f1']})
            row[prefix+'_alta_sem_alerta']=s['alta_sem_alerta']
            row[prefix+'_alta_ref']=s['alta_ref']
        row.update(entry['checks']);rows.append(row)
    return rows


def learning_curves(output, entries, baseline_home):
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,layout='constrained')
    colors={'A_home':'#2264aa','B_home_replay':'#df6b25'}
    initial=next(e for e in entries if e['updates']==0)
    for col,(cohort,title) in enumerate([('home','Home VAL histórica'),('external','Validação externa de seleção')]):
        for arm in colors:
            sequence=[initial]+sorted([e for e in entries if e['arm']==arm],key=lambda e:e['updates'])
            x=[e['updates'] for e in sequence]
            for row,key in enumerate(['precision','recall','alta_sem_alerta']):
                y=[e[cohort][key] if key=='alta_sem_alerta' else 100*e[cohort]['alertas_por_objeto'][key] for e in sequence]
                axes[row,col].plot(x,y,'o-',color=colors[arm],label=arm,linewidth=1.8)
                if cohort=='home':
                    b=baseline_home[key] if key=='alta_sem_alerta' else 100*baseline_home['alertas_por_objeto'][key]
                    if arm=='A_home':axes[row,col].axhline(b,color='#697586',linestyle='--',label='Baseline Home')
        axes[0,col].set_title(title)
        for row,label in enumerate(['Precisão de alertas (%)','Recall de alertas (%)','Objetos de prioridade alta sem alerta']):
            axes[row,col].set_ylabel(label);axes[row,col].grid(alpha=.2)
            low,high=axes[row,col].get_ylim()
            axes[row,col].set_ylim(max(0,low),min(100,high) if row<2 else max(1,high))
        axes[2,col].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[2,col].set_xlim(0,max(1,max(e['updates'] for e in entries)))
        axes[2,col].set_xticks(sorted({e['updates'] for e in entries}))
        axes[2,col].set_xlabel('Atualizações do otimizador por braço')
        axes[0,col].legend(fontsize=8)
    fig.suptitle('Ajustes a partir do candidato 10 · confiança 0,25 · mesmas validações',fontsize=13)
    fig.savefig(Path(output)/'curvas_validacao.png',dpi=160)
    plt.close(fig)


def build_report(output, selection, evaluations, references, cohorts, policy, rule, audit, seed=42):
    output=Path(output)
    rows=checkpoint_rows(selection)
    core.save_csv(output/'comparacao_checkpoints.csv',rows)
    learning_curves(output,selection['assessed'],references['baseline']['home_val_historica']['summary'])
    eligible,exploratory=selection['eligible'],selection['exploratory']
    chosen=eligible or exploratory
    if chosen is None:raise RuntimeError('Candidato inicial deve continuar disponível para inspeção.')
    final=evaluations[chosen['id']]
    bootstrap=[];gallery=[];cards=[];diagnostic=[]
    for cohort,records in cohorts.items():
        for reference_name, reference in references.items():
            for row in r11.paired_bootstrap(reference[cohort]['images'],final[cohort]['images'],1000,seed):
                bootstrap.append({'coorte':cohort,'referencia':reference_name,'checkpoint':chosen['id'],**row})
        pair={'antigo':references['baseline'][cohort],'candidato':final[cohort]}
        diagnostic.extend(r11.diagnosis_rows({cohort:pair}))
        caches={k:r11.load_prediction_cache(v['cache_file']) for k,v in pair.items()}
        mapping={r['id']:r for r in records}
        cases=r11.select_gallery_cases(pair['antigo']['images'],pair['candidato']['images'],limit=12,seed=seed)
        for i,case in enumerate(cases):
            record=mapping[case['id']];relative=f'galeria/{cohort}_{i:02}.jpg'
            r11.render_case(record,caches['antigo'][case['id']],caches['candidato'][case['id']],policy,output/relative)
            gallery.append({**case,'coorte':cohort,'checkpoint':chosen['id'],'arquivo':relative,'creditos':record.get('creditos',{})})
            cards.append('<article><h3>'+html.escape(cohort+' / '+case['categoria'])+'</h3><img loading="lazy" src="'+relative+'" alt="Referência, baseline e checkpoint"><p>Antigo = baseline. Candidato = '+html.escape(chosen['id'])+'.</p></article>')
    core.save_csv(output/'bootstrap_pareado.csv',bootstrap)
    core.save_csv(output/'diagnostico_erros.csv',diagnostic)
    core.save_json(output/'indice_galeria.json',gallery)
    core.save_json(output/'selecao_checkpoints.json',selection)
    status=('Um checkpoint cumpriu todos os requisitos nas validações: '+eligible['id']+'. Requer avaliação futura independente antes de substituir o modelo em uso.' if eligible else 'Nenhum checkpoint cumpriu todos os requisitos. O baseline continua sendo a referência em uso. O melhor resultado exploratório é apresentado para diagnóstico.')
    header=['Checkpoint','Atualizações','P Home','R Home','Alta omitida Home','P externa','R externa','Alta omitida externa','Elegível']
    table='<table><tr>'+''.join('<th>'+x+'</th>' for x in header)+'</tr>'
    for r in rows:
        values=[r['id'],r['updates'],f"{100*r['home_precision']:.2f}%",f"{100*r['home_recall']:.2f}%",r['home_alta_sem_alerta'],f"{100*r['externa_precision']:.2f}%",f"{100*r['externa_recall']:.2f}%",r['externa_alta_sem_alerta'],'Sim' if r['elegivel'] else 'Não']
        table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    table+='</table>'
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TCC — Experimento 12</title><style>body{font:16px/1.6 system-ui;color:#172335;background:#f3f5f8;margin:0}main{max-width:1400px;margin:auto;padding:30px}article,.note{background:white;border:1px solid #dbe2ec;border-radius:10px;padding:18px;margin:20px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;text-align:left;border:1px solid #dbe2ec}.scroll{overflow:auto}h1{line-height:1.2}</style><main><h1>Experimento 12 — Recuperar Home e preservar o ganho externo</h1>'''
    page+='<div class="note"><strong>'+html.escape(status)+'</strong><p>Dois ajustes a partir do mesmo candidato 10. Orçamento de atualizações igual, exposição por fonte diferente e registrada. Política V2 e confiança 0,25 preservadas. As 300 imagens reservadas, Home TEST e Open Images TEST não foram inferidas.</p></div>'
    page+='<h2>Seleção em duas validações</h2><div class="scroll">'+table+'</div><img src="curvas_validacao.png" alt="Métricas nas duas validações por atualizações">'
    page+='<p>Requisitos Home: precisão maior que '+f"{100*rule['home']['precision_maior_que']:.3f}%"+', recall de pelo menos '+f"{100*rule['home']['recall_minimo']:.3f}%"+' e até '+str(rule['home']['alta_sem_alerta_maximo'])+' objetos de prioridade alta sem alerta. Requisitos externos: preservar as três métricas do candidato inicial. As referências foram medidas no mesmo ambiente antes dos novos treinos.</p>'
    page+='<h2>Limites da conclusão</h2><p>A validação externa passou a participar da seleção e deixou de ser uma avaliação final independente deste ajuste. Home VAL é histórica; a origem do primeiro treinamento continua '+html.escape(audit['baseline_original']['estado_proveniencia_original'])+'. Bootstrap sobre as validações é descritivo e não corrige seleção de checkpoints, múltiplas comparações ou agrupamento de imagens da mesma cena. Uma única semente e o orçamento curto não permitem concluir qual abordagem é superior em geral. Open Images tem anotações parciais e não garante cenas internas. Alertas medem concordância com a política nas caixas, sem validação de risco físico.</p>'
    page+='<p><a href="revisao_rotulos.html">Revisar guarda-roupas nos dados de treino</a>. O experimento mantém os rótulos originais. Uma divergência semântica confirmada exige uma correção documentada e um novo experimento.</p><h2>Casos para inspeção</h2><p>Checkpoint mostrado: '+html.escape(chosen['id'])+'. A galeria prioriza diferenças e erros e não estima sua frequência.</p>'+''.join(cards)+'</main></html>'
    (output/'relatorio_experimento_12.html').write_text(page,encoding='utf-8')
    (output/'LEIA_RESULTADO.md').write_text('# Experimento 12\n\n'+status+'\n\nCheckpoint na galeria: '+chosen['id']+'.\n\nAbra relatorio_experimento_12.html e comparacao_checkpoints.csv. Os intervalos em bootstrap_pareado.csv são descritivos após seleção. Nenhum modelo em uso foi substituído. A reserva de 300 imagens permanece sem inferência.\n',encoding='utf-8')
    return chosen,status


def package_reports(output):
    output=Path(output);archive=output/'pacote_experimento_12.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in sorted(output.rglob('*')):
            relative=path.relative_to(output)
            if (not path.is_file() or path==archive or path.name.endswith('.tmp')
                    or path.suffix in {'.pt','.pkl'} or relative.parts[0]=='treinos'
                    or path.name=='predicoes_conf_0001.jsonl.gz'):
                continue
            z.write(path,relative)
    return archive
