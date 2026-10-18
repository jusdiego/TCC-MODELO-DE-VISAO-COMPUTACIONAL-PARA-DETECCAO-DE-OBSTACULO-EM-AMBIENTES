"""Protocolo 17: dois limiares, uma escolha e uma verificação interna."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import diag11_core as core
import diag11_data as data11

VERSION = '17-limiares-chair-table-1.0.0'
COHORTS = ('home_val_historica', 'openimages_val_nova')
PARTITIONS = ('calibracao', 'verificacao_interna')
GRID = (.10, .15, .20, .25, .30, .35, .40)
TUNED_IDS = (2, 3)
FIXED = .25
SPLIT_SEED = '17-grupos-42'
MIN_IMAGES = 50
MIN_ALERT_GT = 30
FIELDS = ('tp_alerta', 'fn_alerta', 'fp_alerta', 'alta_ref', 'alta_sem_alerta',
          'tp_detector', 'fn_detector', 'fp_detector', 'gt_avaliaveis', 'gt_ignorados')
SPEC = {'version': VERSION, 'model': 'candidato_10', 'tuned_class_ids': list(TUNED_IDS),
        'threshold_grid': list(GRID), 'fixed_threshold_other_classes': FIXED,
        'configurations': 49, 'split_seed': SPLIT_SEED,
        'split': 'grupos SHA256/pHash64 distancia<=4; hash do grupo, sem previsoes',
        'split_fraction_target': .5, 'min_images_per_cohort_partition': MIN_IMAGES,
        'min_alert_objects_per_tuned_class_cohort_partition': MIN_ALERT_GT,
        'selection': 'somente calibracao; requisitos relativos as referencias do mesmo subconjunto',
        'ranking': 'precisao Home; recall Home; menos altas omitidas Home; F1 externa; menos classes alteradas; distancia de 0.25; limiares',
        'verification': 'uma configuracao congelada; sem segunda escolha se falhar',
        'full_validation': 'descritiva, apos a escolha; seis limites numericos originais',
        'preexposure': 'ambas as validacoes ja influenciaram o desenvolvimento',
        'training': False, 'test_inference': False, 'reserved_inference': False}

def canonical(value):
    return json.dumps(core.json_clean(value), sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def profile(chair=.25, table=.25):
    if chair not in GRID or table not in GRID:
        raise ValueError('Limiar fora da grade congelada.')
    return {cid: float(chair if cid == 2 else table if cid == 3 else FIXED)
            for cid in range(len(core.NAMES))}

def validate_profile(thresholds):
    normalized = {int(k): v for k, v in thresholds.items()}
    if set(normalized) != set(range(12)):
        raise ValueError('Perfil precisa das doze classes.')
    if normalized != profile(normalized[2], normalized[3]):
        raise ValueError('Somente cadeira e mesa podem variar na grade congelada.')
    return normalized

def profile_id(thresholds):
    p = validate_profile(thresholds)
    return f'chair_{p[2]:.2f}_table_{p[3]:.2f}'.replace('.', 'p')

def group_split(cohorts, output=None):
    """Componentes transitivos de imagens; nenhuma previsão entra na divisão."""
    records = sorted([r for c in COHORTS for r in cohorts[c]], key=lambda r: (r['coorte'], r['id']))
    if len({(r['coorte'], r['id']) for r in records}) != len(records):
        raise ValueError('IDs repetidos na divisão.')
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    phashes = [(int(r['phash'], 16) if isinstance(r['phash'], str) else int(r['phash']))
               if 'phash' in r else data11.phash(r['image']) for r in records]
    links = 0
    for i, r in enumerate(records):
        for j in range(i):
            if r['sha256'] == records[j]['sha256'] or (phashes[i] ^ phashes[j]).bit_count() <= 4:
                union(i, j)
                links += 1
    components = defaultdict(list)
    for i in range(len(records)):
        components[find(i)].append(i)
    rows = []
    assigned = {p: {c: [] for c in COHORTS} for p in PARTITIONS}
    for indices in components.values():
        group = digest(sorted({records[i]['sha256'] for i in indices}))
        bit = int(hashlib.sha256((SPLIT_SEED + '|' + group).encode()).hexdigest(), 16) % 2
        partition = PARTITIONS[bit]
        for i in indices:
            r = {**records[i], 'partition': partition, 'group': group, 'phash': phashes[i]}
            assigned[partition][r['coorte']].append(r)
            rows.append({'coorte': r['coorte'], 'id': r['id'], 'sha256': r['sha256'],
                         'phash': f'{phashes[i]:016x}', 'group': group, 'partition': partition})
    rows.sort(key=lambda r: (r['coorte'], r['id']))
    for p in PARTITIONS:
        for c in COHORTS:
            assigned[p][c].sort(key=lambda r: r['id'])
    counts = {p: {c: len(assigned[p][c]) for c in COHORTS} for p in PARTITIONS}
    audit = {'version': VERSION, 'seed': SPLIT_SEED, 'group_count': len(components),
             'similarity_links': links, 'counts': counts, 'split_sha256': digest(rows),
             'uses_predictions': False, 'verification_is_independent_final_test': False}
    if output:
        core.save_csv(Path(output) / 'divisao_calibracao_verificacao.csv', rows)
        core.save_json(Path(output) / 'auditoria_divisao.json', audit)
    return assigned, rows, audit

def check_split_support(partitions, minimum_images=MIN_IMAGES, minimum_alert_gt=MIN_ALERT_GT, policy=None):
    if policy is None:
        raise ValueError('Política V2 necessária para conferir o suporte.')
    rows = []
    for partition in PARTITIONS:
        for cohort in COHORTS:
            records = partitions[partition][cohort]
            if len(records) < minimum_images:
                raise RuntimeError('Divisão com poucas imagens: ' + partition + '/' + cohort)
            supports = Counter()
            for r in records:
                for g in core.with_policy([g for g in r['gt'] if r['known_mask'] & (1 << g['classe_id'])], r['width'], r['height'], policy):
                    if core.alert_on(g):
                        supports[g['classe_id']] += 1
            for cid in TUNED_IDS:
                n = supports[cid]
                rows.append({'partition': partition, 'coorte': cohort, 'classe_id': cid,
                             'classe': core.NAMES[cid], 'objetos_alerta_ref': n})
                if n < minimum_alert_gt:
                    raise RuntimeError('Suporte insuficiente; a divisão não será refeita: ' + str(rows[-1]))
    return rows

def rule_from_references(baseline_home, candidate_external):
    h, e = baseline_home['alertas_por_objeto'], candidate_external['alertas_por_objeto']
    for s in [h, e]:
        if any(s[k] is None or not math.isfinite(s[k]) for k in ['precision', 'recall', 'f1']):
            raise ValueError('Referência sem suporte para os critérios.')
    return {'home': {'precision_maior_que': h['precision'], 'recall_minimo': h['recall'],
                     'alta_sem_alerta_maximo': baseline_home['alta_sem_alerta']},
            'externa': {'precision_minima': e['precision'], 'recall_minimo': e['recall'],
                        'alta_sem_alerta_maximo': candidate_external['alta_sem_alerta']},
            'origem': 'referencias em 0.25 nos mesmos IDs; regra proporcional e contagens locais'}

def judge(home, external, rule):
    h, e = home['alertas_por_objeto'], external['alertas_por_objeto']
    valid = all(s[k] is not None and math.isfinite(s[k]) for s in [h, e] for k in ['precision', 'recall', 'f1'])
    if not valid:
        return {'elegivel': False, 'metricas_com_suporte': False}
    result = {'home_precision_superior': h['precision'] > rule['home']['precision_maior_que'] + 1e-12,
              'home_recall_preservado': h['recall'] >= rule['home']['recall_minimo'] - 1e-12,
              'home_altas_preservadas': home['alta_sem_alerta'] <= rule['home']['alta_sem_alerta_maximo'],
              'externa_precision_preservada': e['precision'] >= rule['externa']['precision_minima'] - 1e-12,
              'externa_recall_preservado': e['recall'] >= rule['externa']['recall_minimo'] - 1e-12,
              'externa_altas_preservadas': external['alta_sem_alerta'] <= rule['externa']['alta_sem_alerta_maximo']}
    return {**result, 'elegivel': all(result.values()), 'metricas_com_suporte': True}

def choose(calibration_grid):
    if any(r['partition'] != 'calibracao' for r in calibration_grid):
        raise ValueError('Seleção recebeu resultados de verificação.')
    if len(calibration_grid) != 49 or len({r['id'] for r in calibration_grid}) != 49:
        raise ValueError('A grade precisa de exatamente 49 configurações distintas.')
    good = [r for r in calibration_grid if r['checks']['elegivel']]
    def key(r):
        p = validate_profile(r['thresholds'])
        h, e = r['home']['alertas_por_objeto'], r['external']['alertas_por_objeto']
        return (-h['precision'], -h['recall'], r['home']['alta_sem_alerta'], -e['f1'],
                sum(p[c] != .25 for c in TUNED_IDS), sum(abs(p[c] - .25) for c in TUNED_IDS), p[2], p[3])
    selected = min(good, key=key) if good else None
    return {'selected': selected, 'eligible_count': len(good), 'configurations': 49,
            'used_partition': 'calibracao', 'verification_used_for_selection': False,
            'status': 'configuracao_congelada_para_verificacao' if selected else 'nenhuma_configuracao_elegivel_na_calibracao'}

def immutable_json(path, value):
    path = Path(path)
    if path.is_file():
        if digest(json.loads(path.read_text())) != digest(value):
            raise RuntimeError('Registro congelado diverge; não será sobrescrito: ' + path.name)
    else:
        core.save_json(path, value)
    return digest(value)
