"""Galeria local com a mesma imagem, GT e previsões dos dois modelos."""
from __future__ import annotations
import base64,hashlib,html,json,mimetypes,zipfile
from pathlib import Path
from PIL import Image
import diag11_core as core
import cons19_protocol as protocol

CATEGORY_LABELS={'mais_tp_candidato10':'Candidato 10 recupera mais alertas',
    'mais_tp_baseline':'Baseline recupera mais alertas','omissao_alta':'Omissão de alta prioridade',
    'falso_alerta':'Presença de falso alerta','acerto_sem_erro':'Acertos sem FN/FP nas classes avaliáveis',
    'sem_alerta_referencia':'Sem alerta de referência nas classes avaliáveis','complemento':'Complemento por ordem fixa'}

def select(records,evaluations,spec=protocol.SPEC):
    selected=[];manifest=[]
    for cohort in protocol.COHORTS:
        mapping={m:{r['id']:r for r in evaluations[m][cohort]['images']} for m in protocol.MODELS}
        candidates=[r for r in records if r['coorte']==cohort]
        if any(r['role'] not in {'home_val','val_nova'} for r in candidates):raise ValueError('Papel protegido na galeria.')
        used=set();chosen=[]
        def eligible(category,r):
            a,b=[mapping[m][r['id']] for m in protocol.MODELS]
            if category=='mais_tp_candidato10':return b['tp_alerta']>a['tp_alerta']
            if category=='mais_tp_baseline':return a['tp_alerta']>b['tp_alerta']
            if category=='omissao_alta':return max(a['alta_sem_alerta'],b['alta_sem_alerta'])>0
            if category=='falso_alerta':return max(a['fp_alerta'],b['fp_alerta'])>0
            if category=='acerto_sem_erro':return all(v['tp_alerta']>0 and v['fn_alerta']==v['fp_alerta']==0 for v in [a,b])
            if category=='sem_alerta_referencia':return a['tp_alerta']+a['fn_alerta']==0
            return True
        order=lambda r:hashlib.sha256((spec['demo_seed']+'|'+r['id']).encode()).hexdigest()
        for category in list(spec['demo_categories'])+['complemento']:
            limit=spec['demo_images_per_cohort']-len(chosen) if category=='complemento' else spec['demo_per_category']
            pool=sorted([r for r in candidates if r['id'] not in used and eligible(category,r)],key=order)
            for record in pool[:max(0,limit)]:
                if len(chosen)>=spec['demo_images_per_cohort']:break
                used.add(record['id']);chosen.append(record)
                manifest.append({'coorte':cohort,'id':record['id'],'image_sha256':record['sha256'],
                    'categoria':category,'ordem_hash':order(record),'ilustrativa_sem_peso_nas_metricas':True})
        selected+=chosen
    if len({r['id'] for r in selected})!=len(selected):raise ValueError('Imagem repetida na seleção.')
    return selected,manifest

def image_member(record):
    if record['role']=='home_val' and record['coorte']=='home_val_historica':
        name=record['id'].removeprefix('home:')
        if not name.startswith('homeobjects_yolo/images/val/raw_train/'):raise ValueError('Imagem Home fora de VAL.')
        return name
    if record['role']=='val_nova' and record['coorte']=='openimages_val_nova':
        name=record['name']
        if Path(name).name!=name:raise ValueError('Nome externo inválido.')
        return 'images/val_nova/'+name
    raise ValueError('A galeria só aceita as validações de desenvolvimento.')

def extract_images(selected,archives,expected,output):
    output=Path(output);paths={};receipts=[]
    for cohort in protocol.COHORTS:
        rows=[r for r in selected if r['coorte']==cohort]
        if not rows:continue
        archive=Path(archives[cohort]);key='home_zip' if cohort=='home_val_historica' else 'zip_novo'
        if core.sha256_file(archive)!=expected[key]:raise ValueError('ZIP de imagens alterado: '+cohort)
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('ZIP com membros repetidos.')
            for name in z.namelist():core.safe_member(name)
            for r in rows:
                member=image_member(r);core.safe_member(member)
                raw=z.read(member)
                if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Imagem da galeria difere do cache: '+r['id'])
                ext=Path(member).suffix.lower();path=output/'demonstracao/imagens'/(hashlib.sha256(r['id'].encode()).hexdigest()[:24]+ext)
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                with Image.open(path) as image:
                    if image.size!=(r['width'],r['height']):raise ValueError('Dimensões da galeria divergentes.')
                    image.verify()
                paths[r['id']]=path
                receipts.append({'id':r['id'],'role':r['role'],'member':member,'sha256':r['sha256'],'relative':str(path.relative_to(output))})
    if set(paths)!={r['id'] for r in selected}:raise ValueError('Galeria sem alguma imagem.')
    core.save_json(output/'imagens_demo_extraidas.json',receipts)
    return paths

def box_group(obj,record,known=True):
    x1,y1,x2,y2=map(float,obj['bbox']);w,h=record['width'],record['height']
    color='#b91c1c' if obj['prioridade']=='alto' else '#a16207' if obj['prioridade']=='medio' else '#526174'
    if not known:color='#2563eb'
    label=obj['classe']+' · '+obj['prioridade']
    if 'confianca' in obj:label+=f" · {obj['confianca']:.2f}"
    if not known:label+=' · sem cobertura'
    font=max(12,w*.035);stroke=max(2,w*.003)
    tx=max(0,min(x1,w-font));ty=max(font,min(y1,h-font*.2))
    cls='' if core.alert_on(obj) else 'nonalert'
    dash=f' stroke-dasharray="{stroke*3:g} {stroke*2:g}"' if not known else ''
    return f'<g class="box {cls}"><title>{html.escape(label)}</title><rect x="{x1:g}" y="{y1:g}" width="{max(0,x2-x1):g}" height="{max(0,y2-y1):g}" fill="none" stroke="{color}" stroke-width="{stroke:g}"{dash}/><text x="{tx:g}" y="{ty:g}" fill="{color}" stroke="white" stroke-width="{font*.09:g}" paint-order="stroke" font-size="{font:g}" font-family="sans-serif">{html.escape(label)}</text></g>'

def gallery(selected,selection,paths,bundle,evaluations,policy,output):
    output=Path(output);definitions=[];cards=[];summaries=[]
    selection_by_id={r['id']:r for r in selection}
    image_rows={m:{c:{r['id']:r for r in evaluations[m][c]['images']} for c in protocol.COHORTS} for m in protocol.MODELS}
    for index,r in enumerate(selected,1):
        if r['role'] not in {'home_val','val_nova'}:raise ValueError('Papel protegido na renderização.')
        file=Path(paths[r['id']]);raw=file.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Imagem mudou antes de renderizar.')
        token='foto'+str(index);mime=mimetypes.guess_type(file.name)[0] or 'image/jpeg'
        definitions.append(f'<image id="{token}" href="data:{mime};base64,{base64.b64encode(raw).decode()}" width="{r["width"]}" height="{r["height"]}"/>')
        category=selection_by_id[r['id']]['categoria'];panels=[]
        gt=core.with_policy([g for g in r['gt'] if r['known_mask']&(1<<g['classe_id'])],r['width'],r['height'],policy)
        views=[('Referência anotada',gt,None)]
        for model in protocol.MODELS:
            preds=bundle['predictions'][model][r['coorte']][r['id']]
            views.append(('Baseline' if model=='baseline' else 'Candidato 10',core.with_policy(preds,r['width'],r['height'],policy),model))
        for title,objects,model in views:
            boxes=''.join(box_group(o,r,bool(r['known_mask']&(1<<o['classe_id']))) for o in objects)
            svg=f'<svg viewBox="0 0 {r["width"]} {r["height"]}" role="img" aria-label="{html.escape(title)} na imagem {index}"><use href="#{token}"/>{boxes}</svg>'
            if model:
                row=image_rows[model][r['coorte']][r['id']]
                messages=[]
                for obj in objects:
                    if core.alert_on(obj):
                        message=policy.gerar_mensagem(obj,'medio')
                        if not r['known_mask']&(1<<obj['classe_id']):message+=' Classe sem cobertura de referência nesta imagem.'
                        messages.append(message)
                message_html='<ul>'+''.join('<li>'+html.escape(m)+'</li>' for m in messages)+'</ul>' if messages else '<p>Nenhum alerta V2 neste limiar. Isso não atesta ausência de obstáculos.</p>'
                content=f'<p class="counts">TP {row["tp_alerta"]} · FN {row["fn_alerta"]} · FP {row["fp_alerta"]} · omissões altas {row["alta_sem_alerta"]}</p>'+message_html
                summaries.append({'id':r['id'],'coorte':r['coorte'],'modelo':model,'categoria':category,
                    'mensagens':messages,'tp_alerta':row['tp_alerta'],'fn_alerta':row['fn_alerta'],'fp_alerta':row['fp_alerta'],
                    'alta_sem_alerta':row['alta_sem_alerta'],'predicoes':objects})
            else:content='<p>Caixas anotadas nas classes com cobertura. Azul tracejado nas previsões indica classe sem cobertura; seu acerto/erro não é julgado.</p>'
            panels.append('<section class="panel"><h3>'+title+'</h3>'+svg+content+'</section>')
        cohort_label='Home VAL' if r['coorte']==protocol.COHORTS[0] else 'Open Images VAL'
        cards.append(f'<article data-cohort="{r["coorte"]}"><h2>{index:02d} · {cohort_label} · {html.escape(CATEGORY_LABELS[category])}</h2><p class="id">{html.escape(r["id"])}</p><div class="panels">'+''.join(panels)+'</div></article>')
    page='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Demonstração 19 — objetos e alertas V2</title><style>body{font:16px/1.55 system-ui;margin:0;background:#eef2f6;color:#162739}main{max-width:1550px;margin:auto;padding:24px}article,.intro{background:white;border:1px solid #d3dce5;border-radius:12px;padding:20px;margin:18px 0}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.panel{min-width:0}svg{width:100%;max-height:480px;background:#f1f3f5}.id{overflow-wrap:anywhere;color:#526174}.counts{font-weight:600}h1,h2,h3{line-height:1.25}h2{font-size:22px}h3{font-size:18px}.nonalert{display:none}.showall .nonalert{display:inline}button{padding:9px 14px;margin:4px;border:1px solid #8297ac;border-radius:6px;background:white;cursor:pointer}button:focus-visible{outline:3px solid #2563eb}li{margin:8px 0}@media(max-width:1000px){.panels{grid-template-columns:1fr}main{padding:10px}}[hidden]{display:none!important}</style><main><p>TCC · Demonstração em imagens de desenvolvimento</p><h1>Reconhecimento de objetos e mensagens da política V2</h1><div class="intro"><p>Mesma imagem, mesma confiança 0,25: referência anotada, baseline e candidato 10. As previsões foram capturadas anteriormente; esta página não executa inferência em novas imagens.</p><p>Galeria ilustrativa com acertos e erros escolhidos por regras e semente fixas. As métricas completas usam 456 Home VAL e 1.000 OI VAL; esta seleção não mede desempenho.</p><p>Vermelho: prioridade alta V2. Ocre: média. Azul tracejado: classe sem cobertura de anotação, sem julgamento de acerto/erro. A classificação usa posição e tamanho aparentes; não mede distância nem comprova risco físico.</p><p><label><input type="checkbox" onchange="document.body.classList.toggle('showall',this.checked)"> Mostrar também objetos que não geram alerta V2</label></p><nav aria-label="Filtrar imagens"><button type="button" onclick="filtrar('todos')">Todas</button><button type="button" onclick="filtrar('home_val_historica')">Home</button><button type="button" onclick="filtrar('openimages_val_nova')">Open Images</button></nav></div>'''
    page+='<svg width="0" height="0" aria-hidden="true" style="position:absolute"><defs>'+''.join(definitions)+'</defs></svg>'
    page+=''.join(cards)+'''<p>Uso: demonstração acadêmica controlada. Reserva e testes não foram usados. Nenhum modelo foi substituído.</p></main><script>function filtrar(c){document.querySelectorAll('article').forEach(a=>a.hidden=c!=='todos'&&a.dataset.cohort!==c)}</script></html>'''
    path=output/'demonstracao_19.html';path.write_text(page,encoding='utf-8')
    core.save_json(output/'demonstracao_predicoes_e_mensagens.json',summaries)
    return path
