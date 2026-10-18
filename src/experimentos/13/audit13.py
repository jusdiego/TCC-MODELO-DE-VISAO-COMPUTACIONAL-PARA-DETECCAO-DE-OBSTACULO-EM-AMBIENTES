"""Auditoria 13: dados imutáveis, casos de treino/validação e revisão portátil."""
from __future__ import annotations
import base64
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
from PIL import Image,ImageDraw
import numpy as np
import diag11_core as core
from diag11_reports import font
from exp12_protocol import order_id

TRAIN_ROLES={'home_train','train'}
VAL_ROLES={'home_val','val_nova'}
VERSION='13-auditoria-1.0'
REVIEW_VERSION='13-revisoes-1.0'
FIELDS={
 'rotulo':('Rótulo da caixa destacada',[('coerente','Coerente'),('classe_inadequada','Classe inadequada'),('duvidoso','Duvidoso')]),
 'caixa':('Caixa de referência',[('adequada','Adequada'),('parcial','Parcial / imprecisa'),('inadequada','Inadequada'),('duvidoso','Duvidosa')]),
 'ambiente':('Ambiente',[('interno','Interno'),('externo','Externo'),('ambiguo','Ambíguo')]),
 'tipo_imagem':('Tipo de imagem',[('foto','Fotografia'),('render_ilustracao','Render ou ilustração'),('duvidoso','Duvidoso')]),
 'conceito':('O que há na caixa destacada?',[(x,x.replace('_',' ').capitalize()) for x in ['guarda_roupa','closet','aparador','rack','estante','armario_generico','arara_de_roupas','mesa','cadeira','outro','varios_conceitos','duvidoso']]),
 'causa':('Diagnóstico visual do erro',[('rotulo_ou_gt_duvidoso','Rótulo ou caixa de referência duvidosos'),('caixa_previsao','Caixa da previsão desalinhada'),('pequeno_ou_ocluido','Objeto pequeno ou ocluído'),('indicio_confianca_csv','CSV indica detecção abaixo de 0,25'),('omissao_sem_causa_confirmada','Omissão sem causa confirmada'),('sem_erro_no_foco','Sem erro no objeto em foco'),('duvidoso','Duvidoso'),('nao_aplica_treino','Não se aplica: caso de treino')]),
 'destino_treino':('Triagem para um recorte futuro',[('manter_candidato','Candidato a revisão completa'),('revisar_anotacao','Revisar anotação antes de considerar'),('excluir_candidato','Excluir do recorte proposto'),('nao_aplica_validacao','Não se aplica: validação')]),
}

def json_rows(z,name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))

def bool_value(x):
    if isinstance(x,bool):return x
    if x in {'True','False'}:return x=='True'
    raise ValueError('Booleano inválido nos eventos.')

def parse_events(rows):
    out=[]
    for r in rows:
        v=dict(r);v['classe_id']=int(r['classe_id'])
        if not 0<=v['classe_id']<12:raise ValueError('Classe inválida nos eventos.')
        for k in ['bbox_ref','bbox_pred']:
            v[k]=json.loads(r[k]) if r[k] else None
            if v[k] is not None and (len(v[k])!=4 or not np.isfinite(v[k]).all()):raise ValueError('Caixa inválida.')
        for k in ['ha_alerta_ref','ha_alerta_pred']:v[k]=bool_value(r[k])
        v['confianca']=float(r['confianca']) if r['confianca'] else None
        out.append(v)
    return out

def load_source12(path,expected_sha,expected_code):
    if core.sha256_file(path)!=expected_sha:raise RuntimeError('O pacote 12 não é o recebido e analisado.')
    result={'events':{},'images':{}}
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Membros repetidos no pacote 12.')
        for n in z.namelist():core.safe_member(n)
        if z.testzip() is not None:raise RuntimeError('Falha de CRC no pacote 12.')
        result['summary']=json.loads(z.read('resumo_experimento_12.json'))
        result['config']=json.loads(z.read('configuracao_experimento_12.json'))
        result['replay']=json.loads(z.read('manifesto_replay.json'))
        if not result['summary']['concluido'] or result['summary']['reserva_futura_inferida']:
            raise RuntimeError('Execução 12 incompatível com esta auditoria.')
        if result['config']['codigo_embutido_sha256']!=expected_code:raise RuntimeError('Código do 12 diferente do preparado.')
        for n,sha in expected_code.items():
            if hashlib.sha256(z.read('codigo/'+n)).hexdigest()!=sha:raise RuntimeError('Código alterado no pacote 12.')
        for cohort in ['home_val_historica','openimages_val_nova']:
            result['events'][cohort]={};result['images'][cohort]={}
            for model,prefix in [('candidato_10','referencias/candidato_10'),('B_e03','avaliacoes/B_home_replay_e03')]:
                base=prefix+'/'+cohort+'/'
                metrics=json.loads(z.read(base+'metricas.json'))
                images=json_rows(z,base+'alertas_por_imagem.csv')
                for r in images:
                    for k in ['tp_alerta','fn_alerta','fp_alerta','alta_ref','alta_sem_alerta']:r[k]=int(r[k])
                if len(images)!=len({r['id'] for r in images}) or len(images)!=metrics['imagens']:
                    raise RuntimeError('IDs ou contagens inconsistentes nos registros do 12.')
                for k,v in [('tp_alerta',metrics['alertas_por_objeto']['tp']),('fn_alerta',metrics['alertas_por_objeto']['fn']),('fp_alerta',metrics['alertas_por_objeto']['fp']),('alta_ref',metrics['alta_ref']),('alta_sem_alerta',metrics['alta_sem_alerta'])]:
                    if sum(r[k] for r in images)!=v:raise RuntimeError('Métricas divergem das contagens por imagem.')
                result['events'][cohort][model]=parse_events(json_rows(z,base+'avaliacao_objetos.csv'))
                result['images'][cohort][model]={r['id']:r for r in images}
            if result['images'][cohort]['candidato_10'].keys()!=result['images'][cohort]['B_e03'].keys():
                raise RuntimeError('As duas versões não avaliaram os mesmos IDs.')
    return result

def event_key(event):
    return (event['id'],event['classe_id'],tuple(round(x,6) for x in event['bbox_ref']))

def error_plan(bundle,seed=42):
    plan=[];receipts=[]
    for cohort,quotas in [
        ('home_val_historica',{'nova_omissao_alta':4,'perda_de_alerta':2,'omissao_persistente':3,'melhora':2,'controle':1}),
        ('openimages_val_nova',{'nova_omissao_alta':14,'perda_de_alerta':3,'omissao_persistente':3,'melhora':2,'controle':2})]:
        old,new=[bundle['events'][cohort][m] for m in ['candidato_10','B_e03']]
        old_gt=defaultdict(list);new_gt=defaultdict(list)
        for r in old:
            if r['bbox_ref'] is not None:old_gt[event_key(r)].append(r)
        for r in new:
            if r['bbox_ref'] is not None:new_gt[event_key(r)].append(r)
        if Counter({k:len(v) for k,v in old_gt.items()})!=Counter({k:len(v) for k,v in new_gt.items()}):
            raise RuntimeError('Referências de objetos diferentes entre as avaliações.')
        pools=defaultdict(list);duplicates=0
        for key,old_rows in old_gt.items():
            if len(old_rows)!=1:duplicates+=1;continue
            a,b=old_rows[0],new_gt[key][0]
            if a['classe_id'] not in {2,3} or not a['ha_alerta_ref']:continue
            if a['ha_alerta_pred'] and not b['ha_alerta_pred']:
                category='nova_omissao_alta' if b['prioridade_ref']=='alto' else 'perda_de_alerta'
            elif not a['ha_alerta_pred'] and not b['ha_alerta_pred']:category='omissao_persistente'
            elif not a['ha_alerta_pred'] and b['ha_alerta_pred']:category='melhora'
            else:category='controle'
            pools[category].append({'coorte':cohort,'old':a,'new':b,'categoria':category,'key':key})
        for group in pools.values():group.sort(key=lambda r:order_id(core.canonical(r['key']),seed))
        chosen=[];seen=set();seen_images=set()
        def take(group,n,unique_images=True):
            added=0
            for item in pools[group]:
                if added>=n:break
                if item['key'] in seen or (unique_images and item['old']['id'] in seen_images):continue
                chosen.append(item);seen.add(item['key']);seen_images.add(item['old']['id']);added+=1
            return added
        for group,n in quotas.items():take(group,n)
        for distinct in [True,False]:
            for group in quotas:take(group,sum(quotas.values())-len(chosen),distinct)
        plan+=chosen
        receipts.append({'coorte':cohort,'limite':sum(quotas.values()),'selecionados':len(chosen),
            'disponiveis_por_categoria':{k:len(v) for k,v in pools.items()},
            'selecionados_por_categoria':dict(Counter(r['categoria'] for r in chosen)),
            'referencias_duplicadas_ambiguas_excluidas':duplicates})
    return plan,receipts

def replay_plan(manifest,history,seed=42):
    rows=manifest['imagens'];byid={r['id']:r for r in history['rows']}
    if len(rows)!=2000 or len({r['id'] for r in rows})!=2000:raise RuntimeError('Replay 12 diferente do congelado.')
    for r in rows:
        prior=byid[r['id']]
        if prior['split']!='train' or prior['imagem_sha256']!=r['sha256'] or prior['lote']!=r['lote']:
            raise RuntimeError('O replay não corresponde ao TRAIN histórico.')
    selected=[]
    for cid,n in [(7,24),(3,6),(2,6)]:
        group=sorted([r for r in rows if int(r['classe_primaria'])==cid],key=lambda r:order_id(r['id'],seed))
        if len(group)<n:raise RuntimeError('Suporte insuficiente na amostra de treino para auditoria.')
        selected += [byid[r['id']] for r in group[:n]]
    return selected

def focus_index(record,cid):
    candidates=[i for i,o in enumerate(record['gt']) if o['classe_id']==cid]
    if not candidates:raise RuntimeError('Classe selecionada sem caixa no label original.')
    return min(candidates,key=lambda i:(-(record['gt'][i]['bbox'][2]-record['gt'][i]['bbox'][0])*(record['gt'][i]['bbox'][3]-record['gt'][i]['bbox'][1]),i))

def make_case(record,index,group,reason,context=None):
    if record['role'] not in TRAIN_ROLES|VAL_ROLES:raise RuntimeError('Reserva ou teste entrou nos casos.')
    obj=record['gt'][index]
    identity=[record['id'],record['role'],record['sha256'],obj['classe_id'],obj['bbox']]
    cid=hashlib.sha256(core.canonical(identity).encode()).hexdigest()[:24]
    return {'case_id':cid,'id':record['id'],'imagem':record['name'],'papel':record['role'],
            'fonte':'Home' if record['role'].startswith('home_') else 'Open Images',
            'coorte':record['role'] if record['role'] in TRAIN_ROLES else record['coorte'],'grupo':group,'motivo':reason,'classe_id':obj['classe_id'],
            'classe':core.NAMES[obj['classe_id']],'gt_index':index,'bbox_ref':obj['bbox'],
            'width':record['width'],'height':record['height'],'known_mask':record['known_mask'],
            'image_sha256':record['sha256'],'label_sha256':record['label_sha256'],
            'arquivo':'galeria/'+cid+'.jpg','contexto':context or {}}

def build_cases(home,replay,val,plan,seed=42):
    cases=[];records={r['id']:r for r in home+list(replay.values())+val}
    for source,items in [('Home',[r for r in home if r['role']=='home_train']),('Open Images',list(replay.values()))]:
        for cid,n in [(7,24),(3,6),(2,6)]:
            candidates=[r for r in items if any(o['classe_id']==cid for o in r['gt'])]
            if source=='Open Images':candidates=[r for r in candidates if int(r['classe_primaria'])==cid]
            candidates.sort(key=lambda r:order_id(r['id'],seed))
            if len(candidates)<n:raise RuntimeError('Dados de treino insuficientes para a auditoria planejada.')
            for r in candidates[:n]:
                cases.append(make_case(r,focus_index(r,cid),'semantica_treino' if cid==7 else 'rotulos_treino',
                    'Amostra por hash de imagens com a classe; maior caixa da classe em foco.'))
    for item in plan:
        r=records[item['old']['id']]
        expected='home_val' if item['coorte']=='home_val_historica' else 'val_nova'
        if r['role']!=expected:raise RuntimeError('Papel inválido para revisão de erros.')
        indices=[i for i,g in enumerate(r['gt']) if g['classe_id']==item['old']['classe_id'] and np.allclose(g['bbox'],item['old']['bbox_ref'],rtol=0,atol=.001)]
        if len(indices)!=1:raise RuntimeError('Objeto do registro não corresponde univocamente ao GT autenticado.')
        cases.append(make_case(r,indices[0],'erros_validacao',item['categoria'],{
            'evento_candidato10':item['old']['evento'],'evento_B_e03':item['new']['evento'],
            'alerta_candidato10':item['old']['ha_alerta_pred'],'alerta_B_e03':item['new']['ha_alerta_pred'],
            'prioridade_ref':item['old']['prioridade_ref'],'indicios_candidato10':item['old']['indicios'],
            'indicios_B_e03':item['new']['indicios']}))
    if len({c['case_id'] for c in cases})!=len(cases):raise RuntimeError('Casos duplicados na amostra.')
    group_order={'semantica_treino':0,'rotulos_treino':1,'erros_validacao':2}
    return sorted(cases,key=lambda c:(group_order[c['grupo']],c['fonte'],c['classe_id'],c['case_id'])),records

def predictions_from_events(events):
    result=defaultdict(list);seen=set()
    for e in events:
        if e['bbox_pred'] is None:continue
        if e['confianca'] is None or e['confianca']<.25:raise RuntimeError('Previsão sem confiança mínima nos registros.')
        key=(e['id'],e['classe_id'],tuple(e['bbox_pred']),e['confianca'])
        if key in seen:continue
        seen.add(key);result[e['id']].append({'classe_id':e['classe_id'],'classe':e['classe'],
            'bbox':e['bbox_pred'],'confianca':e['confianca']})
    return result

def draw_panel(record,objects,title,focus=None,predictions=False):
    canvas=Image.new('RGB',(540,480),'#111c2c');d=ImageDraw.Draw(canvas)
    d.text((12,10),title,fill='white',font=font(15))
    with Image.open(record['image']) as image:
        image=image.convert('RGB');scale=min(520/image.width,390/image.height)
        image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
    x0,y0=(540-image.width)//2,48+(390-image.height)//2;canvas.paste(image,(x0,y0))
    sx,sy=image.width/record['width'],image.height/record['height']
    ordered=[i for i in range(len(objects)) if i!=focus]+([focus] if focus is not None else [])
    for i in ordered:
        o=objects[i];box=[x0+o['bbox'][0]*sx,y0+o['bbox'][1]*sy,x0+o['bbox'][2]*sx,y0+o['bbox'][3]*sy]
        color='#c09bff' if predictions else ('#58e8df' if i==focus else '#9ca9bb')
        d.rectangle(box,outline=color,width=4 if i==focus else 2)
        label=o['classe']+(f" {o['confianca']:.2f}" if predictions else (' / FOCO' if i==focus else ''))
        tx=max(1,min(box[0],460));ty=max(43,box[1]-16)
        d.rectangle(d.textbbox((tx,ty),label,font=font(11)),fill='#111c2c');d.text((tx,ty),label,fill=color,font=font(11))
    d.text((12,454),'Roxo: previsões avaliáveis @0,25' if predictions else 'Ciano: caixa de referência a revisar | cinza: demais rótulos',fill='#d3ddeb',font=font(10))
    return canvas

def render_cases(cases,records,bundle,output):
    output=Path(output);pred={}
    for cohort,models in bundle['events'].items():
        pred[cohort]={m:predictions_from_events(events) for m,events in models.items()}
    for i,case in enumerate(cases,1):
        record=records[case['id']]
        panels=[draw_panel(record,record['gt'],'Rótulos originais / '+case['fonte'],case['gt_index'])]
        if case['papel'] in VAL_ROLES:
            for model,title in [('candidato_10','Candidato 10 / registros do experimento 12'),('B_e03','Home + replay / época 3 do experimento 12')]:
                panels.append(draw_panel(record,pred[case['coorte']][model].get(case['id'],[]),title,predictions=True))
        image=Image.new('RGB',(540*len(panels),480),'#111c2c')
        for n,panel in enumerate(panels):image.paste(panel,(540*n,0))
        path=output/case['arquivo'];path.parent.mkdir(parents=True,exist_ok=True);image.save(path,quality=86)
        if i%20==0 or i==len(cases):print('Casos renderizados:',i,'/',len(cases),flush=True)

def make_manifest(cases,inputs,receipts,seed=42):
    manifest={'versao':VERSION,'inputs':inputs,'seed':seed,'casos':cases,'selecao_erros':receipts,
        'amostra_dirigida_nao_estima_prevalencia':True,'validacao_nunca_vira_treino':True,
        'inferencias_novas':0,'novos_treinamentos':0,'reserva_futura_extraida':False,'reserva_futura_inferida':False,
        'limite_galeria':'Previsões em classes avaliáveis registradas nos CSVs a partir de confiança 0,25. Sem previsões de classes sem cobertura ou de confiança inferior.'}
    manifest['audit_manifest_sha256']=hashlib.sha256(core.canonical(manifest).encode()).hexdigest()
    return manifest

def empty_answer(case):
    answer={'case_id':case['case_id'],'status':'pendente',**{k:'pendente' for k in FIELDS},'observacao':''}
    answer['causa']='nao_aplica_treino' if case['papel'] in TRAIN_ROLES else 'pendente'
    answer['destino_treino']='nao_aplica_validacao' if case['papel'] in VAL_ROLES else 'pendente'
    return answer

def empty_review(manifest):
    return {'versao':REVIEW_VERSION,'audit_manifest_sha256':manifest['audit_manifest_sha256'],
            'responsavel':'','wardrobe_decisao':'pendente','observacao_taxonomia':'',
            'casos':[empty_answer(c) for c in manifest['casos']]}

def validate_manifest(manifest):
    raw=dict(manifest);claimed=raw.pop('audit_manifest_sha256',None)
    if hashlib.sha256(core.canonical(raw).encode()).hexdigest()!=claimed:
        raise ValueError('Manifesto da auditoria alterado.')
    if len({c['case_id'] for c in manifest['casos']})!=len(manifest['casos']):
        raise ValueError('Manifesto com casos repetidos.')
    for c in manifest['casos']:
        if c['papel'] not in TRAIN_ROLES|VAL_ROLES:raise ValueError('Reserva ou teste no manifesto.')
        identity=[c['id'],c['papel'],c['image_sha256'],c['classe_id'],c['bbox_ref']]
        if hashlib.sha256(core.canonical(identity).encode()).hexdigest()[:24]!=c['case_id']:
            raise ValueError('Identidade do caso alterada.')


def validate_review(manifest,review):
    validate_manifest(manifest)
    if review.get('versao')!=REVIEW_VERSION or review.get('audit_manifest_sha256')!=manifest['audit_manifest_sha256']:
        raise ValueError('Revisão pertence a outra amostra ou versão de auditoria.')
    if review.get('wardrobe_decisao') not in {'pendente','guarda_roupa_closet','armazenamento_amplo','separar_classes'}:
        raise ValueError('Decisão de taxonomia inválida.')
    for field in ['responsavel','observacao_taxonomia']:
        if not isinstance(review.get(field),str) or len(review[field])>10000:raise ValueError('Texto de revisão inválido.')
    cases={c['case_id']:c for c in manifest['casos']};answers={}
    for row in review.get('casos',[]):
        cid=row.get('case_id')
        if cid not in cases or cid in answers:raise ValueError('Caso desconhecido ou repetido na revisão.')
        if set(row)!=set(empty_answer(cases[cid])):raise ValueError('Campos de revisão diferentes do formulário.')
        if row['status'] not in {'pendente','revisado'}:raise ValueError('Status de revisão inválido.')
        if not isinstance(row['observacao'],str) or len(row['observacao'])>10000:raise ValueError('Observação inválida.')
        for key,(_,options) in FIELDS.items():
            if row[key] not in {'pendente'}|{x[0] for x in options}:raise ValueError('Resposta fora das opções: '+key)
        case=cases[cid]
        if case['papel'] in VAL_ROLES and row['destino_treino']!='nao_aplica_validacao':
            raise ValueError('Validação não pode ser destinada a treino, mesmo por edição do JSON.')
        if case['papel'] in TRAIN_ROLES:
            if row['causa']!='nao_aplica_treino' or row['destino_treino']=='nao_aplica_validacao':raise ValueError('Papel de treino inconsistente.')
        elif row['causa']=='nao_aplica_treino':raise ValueError('Causa de treino em validação.')
        if row['causa']=='indicio_confianca_csv' and not any('deteccao_abaixo_da_confianca_025' in str(v) for v in case['contexto'].values()):
            raise ValueError('O CSV não registra esse indício de confiança para o caso.')
        if row['status']=='revisado' and (any(row[k]=='pendente' for k in FIELDS) or not review['responsavel'].strip()):
            raise ValueError('Caso marcado revisado sem respostas completas ou responsável.')
        answers[cid]=row
    if set(answers)!=set(cases):raise ValueError('A revisão precisa conter todos os casos, inclusive os pendentes.')
    return answers

def write_review_outputs(manifest,review,output):
    output=Path(output);answers=validate_review(manifest,review);rows=[];triage=[]
    for case in manifest['casos']:
        a=answers[case['case_id']]
        row={k:case[k] for k in ['case_id','id','fonte','papel','classe','image_sha256','label_sha256']}
        row.update(a);rows.append(row)
        if case['papel'] in TRAIN_ROLES:
            eligible=(a['status']=='revisado' and a['rotulo']=='coerente' and a['caixa']=='adequada' and
                a['ambiente']=='interno' and a['tipo_imagem']=='foto' and a['destino_treino']=='manter_candidato')
            if case['classe_id']==7:
                allowed={'guarda_roupa','closet'} if review['wardrobe_decisao']=='guarda_roupa_closet' else {'guarda_roupa','closet','aparador','rack','estante','armario_generico','arara_de_roupas'} if review['wardrobe_decisao']=='armazenamento_amplo' else set()
                eligible=eligible and a['conceito'] in allowed
            triage.append({**row,'candidato_a_revisao_completa':bool(eligible),
                'limite':'Somente a caixa em foco foi revisada; não é uma imagem inteira aprovada nem um dataset pronto.'})
    complete=sum(a['status']=='revisado' for a in answers.values())
    status={'geracao_concluida':True,'casos':len(rows),'casos_revisados':complete,'casos_pendentes':len(rows)-complete,
        'revisao_humana_concluida':complete==len(rows) and review['wardrobe_decisao']!='pendente',
        'decisao_wardrobe':review['wardrobe_decisao'],'objetos_treino_candidatos':sum(r['candidato_a_revisao_completa'] for r in triage),
        'novo_dataset_pronto':False,'rotulos_alterados':False,'novos_treinamentos':0,'inferencias_novas':0,
        'reserva_futura_extraida':False,'reserva_futura_inferida':False,
        'validacao_excluida_de_propostas_treino':True,'audit_manifest_sha256':manifest['audit_manifest_sha256']}
    core.save_csv(output/'revisoes_13.csv',rows)
    core.save_csv(output/'triagem_objetos_treino.csv',triage)
    core.save_json(output/'revisoes_13.json',review)
    core.save_json(output/'resumo_auditoria_13.json',status)
    core.save_json(output/'proposta_taxonomia.json',{'classe_atual':'wardrobe','classe_id':7,
        'decisao':review['wardrobe_decisao'],'observacao':review['observacao_taxonomia'],
        'aplicada':False,'observacao_tecnica':'Ampliar ou separar a classe exige reanotação e nova versão do mapeamento/cobertura. Nenhuma anotação ou validação histórica é corrigida silenciosamente.'})
    return status

def build_reviewer(manifest,review,template,output):
    output=Path(output);validate_review(manifest,review)
    cards=[]
    for case in manifest['casos']:
        raw=(output/case['arquivo']).read_bytes()
        cards.append({**case,'image_data':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()})
    payload={'manifest_sha':manifest['audit_manifest_sha256'],'review':review,'cases':cards,'fields':FIELDS}
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    if template.count('__PAYLOAD__')!=1:raise ValueError('Template do revisor inválido.')
    path=output/'revisor_13.html';path.write_text(template.replace('__PAYLOAD__',encoded),encoding='utf-8')
    return path

def package_reports(output):
    output=Path(output);path=output/'pacote_auditoria_13.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p!=path and not p.name.endswith('.tmp'):
                z.write(p,p.relative_to(output))
    return path


def class_differences(bundle):
    rows=[]
    for cohort,models in bundle['events'].items():
        metrics={}
        for model,events in models.items():
            for cid in range(12):
                es=[e for e in events if e['classe_id']==cid]
                metrics[model,cid]={
                    'tp':sum(e['ha_alerta_ref'] and e['ha_alerta_pred'] for e in es),
                    'fn':sum(e['ha_alerta_ref'] and not e['ha_alerta_pred'] for e in es),
                    'fp':sum(not e['ha_alerta_ref'] and e['ha_alerta_pred'] for e in es),
                    'alta_sem_alerta':sum(e['prioridade_ref']=='alto' and not e['ha_alerta_pred'] for e in es)}
            images=bundle['images'][cohort][model].values()
            for k,col in [('tp','tp_alerta'),('fn','fn_alerta'),('fp','fp_alerta'),('alta_sem_alerta','alta_sem_alerta')]:
                if sum(metrics[model,cid][k] for cid in range(12))!=sum(r[col] for r in images):
                    raise RuntimeError('Eventos não reproduzem as métricas por imagem.')
        for cid in range(12):
            row={'coorte':cohort,'classe':core.NAMES[cid],'classe_id':cid}
            for k in ['tp','fn','fp','alta_sem_alerta']:
                a,b=metrics['candidato_10',cid][k],metrics['B_e03',cid][k]
                row.update({k+'_candidato10':a,k+'_B_e03':b,'delta_'+k:b-a})
            rows.append(row)
    return rows
