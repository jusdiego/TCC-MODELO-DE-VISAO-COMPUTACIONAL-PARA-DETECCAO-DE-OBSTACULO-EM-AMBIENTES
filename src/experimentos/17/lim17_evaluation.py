"""Correspondências refeitas para cada limiar, usando a V2 original."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path

import diag11_core as core
import conf14_evaluation as historical
import lim17_protocol as protocol

def _guard(record):
    expected = {'home_val_historica': 'home_val', 'openimages_val_nova': 'val_nova'}
    if record['coorte'] not in expected or record['role'] != expected[record['coorte']]:
        raise ValueError('Somente as duas validações autorizadas podem ser avaliadas.')

def _event(record, name, kind, gt, pred, overlap, low, thresholds):
    row = core.record_event(record, name, kind, gt, pred, overlap, low)
    cid = int(row['classe_id'])
    threshold = thresholds[cid]
    row['limiar_confianca_classe'] = threshold
    row['melhor_confianca_mesma_classe'] = None
    if kind == 'objeto_nao_detectado':
        excluded = {'deteccao_abaixo_da_confianca_025', 'sobreposicao_com_previsao_de_outra_classe',
                    'omissao_sem_indicio_automatico'}
        flags = [v for v in row['indicios'].split('|') if v and v not in excluded]
        overlaps = core.iou_matrix([gt], low)[0]
        same = [i for i, p in enumerate(low) if p['classe_id'] == cid]
        if same:
            best = max(same, key=lambda i: overlaps[i])
            row['melhor_confianca_mesma_classe'] = low[best]['confianca']
            if overlaps[best] >= .5 and low[best]['confianca'] < threshold:
                flags.append('deteccao_abaixo_do_limiar_da_classe')
        if any(overlaps[i] >= .5 and p['classe_id'] != cid
               and p['confianca'] >= thresholds[p['classe_id']] for i, p in enumerate(low)):
            flags.append('sobreposicao_com_previsao_de_outra_classe')
        row['indicios'] = '|'.join(flags or ['omissao_sem_indicio_automatico'])
    return row

def analyse(record, predictions, thresholds, policy, matcher, name):
    _guard(record)
    p = protocol.validate_profile(thresholds)
    known = int(record['known_mask'])
    gt = core.with_policy([g for g in record['gt'] if known & (1 << g['classe_id'])],
                          record['width'], record['height'], policy)
    low = [v for v in predictions if known & (1 << v['classe_id'])]
    all_pred = core.with_policy([v for v in predictions if v['confianca'] >= p[v['classe_id']]],
                                record['width'], record['height'], policy)
    pred = [v for v in all_pred if known & (1 << v['classe_id'])]
    pairs, missing, extras = matcher(gt, pred)
    events = [_event(record, name, 'correspondencia', gt[gi], pred[pi], float(iou), low, p)
              for gi, pi, iou in pairs]
    events += [_event(record, name, 'objeto_nao_detectado', gt[gi], None, None, low, p) for gi in missing]
    events += [_event(record, name, 'previsao_nao_associada', None, pred[pi], None, low, p) for pi in extras]
    messages = [policy.gerar_mensagem(v, 'medio') for v in all_pred if core.alert_on(v)]
    if any(not isinstance(s, str) or not s.strip() for s in messages):
        raise RuntimeError('Alerta sem mensagem.')
    image = {'coorte': record['coorte'], 'imagem': record['name'], 'id': record['id'], 'modelo': name,
             'partition': record.get('partition', 'completa_descritiva'),
             'perfil_limiares': protocol.profile_id(p),
             'limiar_cadeira': p[2], 'limiar_mesa': p[3],
             'tp_alerta': sum(e['ha_alerta_ref'] and e['ha_alerta_pred'] for e in events),
             'fn_alerta': sum(e['ha_alerta_ref'] and not e['ha_alerta_pred'] for e in events),
             'fp_alerta': sum(not e['ha_alerta_ref'] and e['ha_alerta_pred'] for e in events),
             'alta_ref': sum(g['prioridade'] == 'alto' for g in gt),
             'alta_sem_alerta': sum(e['prioridade_ref'] == 'alto' and not e['ha_alerta_pred'] for e in events),
             'tp_detector': len(pairs), 'fn_detector': len(missing), 'fp_detector': len(extras),
             'gt_avaliaveis': len(gt), 'gt_ignorados': len(record['gt']) - len(gt),
             'predicoes_ignoradas_conf': len(all_pred) - len(pred),
             'cobertura_completa_alertas': known & core.ALERT_MASK == core.ALERT_MASK,
             'ha_alerta_ref': any(core.alert_on(g) for g in gt),
             'ha_alerta_pred': any(core.alert_on(v) for v in pred), 'mensagem_alerta': '\n'.join(messages)}
    return image, events

def evaluate(records, cache, thresholds, policy, matcher, name, output=None):
    p = protocol.validate_profile(thresholds)
    if not records or any(r['id'] not in cache for r in records):
        raise ValueError('Cache incompleto ou avaliação vazia.')
    images, events = [], []
    for record in records:
        im, ev = analyse(record, cache[record['id']], p, policy, matcher, name)
        images.append(im)
        events.extend(ev)
    summary, classes = historical.summarise(images, events, .25)
    summary.update({'modelo': name, 'conf_alertas': .25 if p == protocol.profile() else None,
                    'conf_alertas_por_classe': p, 'perfil_limiares': protocol.profile_id(p),
                    'partition': images[0]['partition'], 'ap_recalculada': False})
    for row in classes:
        row['conf_alertas'] = p[row['classe_id']]
    result = {'summary': summary, 'classes': classes, 'images': images, 'events': events}
    if output:
        output = Path(output)
        core.save_json(output / 'metricas.json', summary)
        core.save_csv(output / 'metricas_por_classe.csv', classes)
        core.save_csv(output / 'alertas_por_imagem.csv', images)
        core.save_csv(output / 'avaliacao_objetos.csv', events)
    return result

def _counts(gt, pred, matcher):
    pairs, missing, extras = matcher(gt, pred)
    states = [(gt[i], pred[j]) for i, j, _ in pairs]
    states += [(gt[i], None) for i in missing]
    states += [(None, pred[j]) for j in extras]
    return {'tp_alerta': sum(core.alert_on(g) and core.alert_on(p) for g, p in states),
            'fn_alerta': sum(core.alert_on(g) and not core.alert_on(p) for g, p in states),
            'fp_alerta': sum(not core.alert_on(g) and core.alert_on(p) for g, p in states),
            'alta_ref': sum(g['prioridade'] == 'alto' for g in gt),
            'alta_sem_alerta': sum(bool(g and g['prioridade'] == 'alto') and not core.alert_on(p) for g, p in states),
            'tp_detector': len(pairs), 'fn_detector': len(missing), 'fp_detector': len(extras),
            'gt_avaliaveis': len(gt), 'gt_ignorados': 0}

def calibration_frontier(records, cache, policy, matcher, progress=None):
    """Pré-calcula por classe: o associador original não cruza classes."""
    if not records or any(r.get('partition') != 'calibracao' for r in records):
        raise ValueError('A grade só pode pré-calcular a partição de calibração.')
    book = []
    for index, r in enumerate(records, 1):
        _guard(r)
        gt = core.with_policy([g for g in r['gt'] if r['known_mask'] & (1 << g['classe_id'])], r['width'], r['height'], policy)
        # A política só recebe propostas que podem entrar na grade; geometrias do cache não são modificadas.
        eligible = [v for v in cache[r['id']] if r['known_mask'] & (1 << v['classe_id'])
                    and v['confianca'] >= (min(protocol.GRID) if v['classe_id'] in protocol.TUNED_IDS else .25)]
        pred = core.with_policy(eligible, r['width'], r['height'], policy)
        fixed = {f: 0 for f in protocol.FIELDS}
        variable = {}
        for cid in range(12):
            gs = [g for g in gt if g['classe_id'] == cid]
            ps = [p for p in pred if p['classe_id'] == cid]
            if cid in protocol.TUNED_IDS:
                variable[str(cid)] = {str(t): _counts(gs, [p for p in ps if p['confianca'] >= t], matcher) for t in protocol.GRID}
            else:
                count = _counts(gs, ps, matcher)
                for f in fixed:
                    fixed[f] += count[f]
        fixed['gt_ignorados'] = len(r['gt']) - len(gt)
        book.append({'id': r['id'], 'coorte': r['coorte'], 'partition': 'calibracao',
                     'fixed': fixed, 'variable': variable})
        if progress and (index % 100 == 0 or index == len(records)):
            progress(f'Fronteira de calibração {r["coorte"]}: {index}/{len(records)}')
    return book

def frontier_summary(book, thresholds):
    p = protocol.validate_profile(thresholds)
    if not book or any(row['partition'] != 'calibracao' for row in book):
        raise ValueError('Fronteira não pertence à calibração.')
    sums = Counter()
    for row in book:
        sums.update(row['fixed'])
        for cid in protocol.TUNED_IDS:
            sums.update(row['variable'][str(cid)][str(p[cid])])
    return {'coorte': book[0]['coorte'], 'imagens': len(book),
            'alertas_por_objeto': core.alert_metrics(sums['tp_alerta'], sums['fn_alerta'], sums['fp_alerta']),
            'alta_ref': sums['alta_ref'], 'alta_sem_alerta': sums['alta_sem_alerta'],
            'detector_no_limiar': core.alert_metrics(sums['tp_detector'], sums['fn_detector'], sums['fp_detector']),
            'gt_ignorados_sem_cobertura': sums['gt_ignorados'], 'partition': 'calibracao'}

def search_grid(books, rule):
    if set(books) != set(protocol.COHORTS):
        raise ValueError('São necessárias as duas coortes de calibração.')
    rows = []
    for chair in protocol.GRID:
        for table in protocol.GRID:
            p = protocol.profile(chair, table)
            home = frontier_summary(books[protocol.COHORTS[0]], p)
            external = frontier_summary(books[protocol.COHORTS[1]], p)
            rows.append({'id': protocol.profile_id(p), 'thresholds': p, 'partition': 'calibracao',
                         'home': home, 'external': external, 'checks': protocol.judge(home, external, rule)})
    return rows

def compare_reference(current, old, cohort, model):
    """Reproduz contadores por imagem e estados de cada objeto GT em 0,25."""
    fields = ['tp_alerta', 'fn_alerta', 'fp_alerta', 'alta_ref', 'alta_sem_alerta',
              'tp_detector', 'fn_detector', 'fp_detector', 'gt_avaliaveis', 'gt_ignorados', 'predicoes_ignoradas_conf']
    a = {x['id']: x for x in old['images']}
    b = {x['id']: x for x in current['images']}
    if a.keys() != b.keys():
        raise ValueError('IDs da referência divergentes.')
    rows = []
    for iid in sorted(a):
        for field in fields:
            ok = int(a[iid][field]) == int(b[iid][field])
            rows.append({'modelo': model, 'coorte': cohort, 'id': iid, 'metrica': field,
                         'valor_16': int(a[iid][field]), 'valor_17': int(b[iid][field]), 'igual': ok})
    def gtmap(events):
        result = {}
        for x in events:
            box = x['bbox_ref']
            if not box:
                continue
            box = json.loads(box) if isinstance(box, str) else box
            k = (x['id'], int(x['classe_id']), tuple(round(float(v), 6) for v in box))
            if k in result:
                raise ValueError('Objeto GT repetido.')
            result[k] = x
        return result
    ga, gb = gtmap(old['events']), gtmap(current['events'])
    if ga.keys() != gb.keys():
        raise ValueError('Objetos GT diferentes da referência.')
    def norm(v):
        return {'True': True, 'False': False}.get(v, v) if isinstance(v, str) else v
    gt_rows = []
    for k in sorted(ga):
        ok = all(norm(ga[k][f]) == norm(gb[k][f]) for f in
                 ['evento', 'prioridade_ref', 'prioridade_pred', 'ha_alerta_ref', 'ha_alerta_pred'])
        gt_rows.append({'modelo': model, 'coorte': cohort, 'id': k[0], 'classe_id': k[1],
                        'bbox_ref': json.dumps(k[2]), 'igual': ok})
    status = {'modelo': model, 'coorte': cohort, 'checagens_imagem': len(rows),
              'divergencias_imagem': sum(not x['igual'] for x in rows), 'objetos_GT': len(gt_rows),
              'divergencias_objetos': sum(not x['igual'] for x in gt_rows)}
    status['reproduziu'] = status['divergencias_imagem'] == status['divergencias_objetos'] == 0
    return status, rows, gt_rows
