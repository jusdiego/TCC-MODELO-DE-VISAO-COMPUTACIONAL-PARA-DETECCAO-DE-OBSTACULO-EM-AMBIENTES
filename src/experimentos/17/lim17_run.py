"""Ordem do experimento: reproduzir, escolher em calibração, congelar, verificar."""
from __future__ import annotations
from collections import Counter
import gzip
import json
from pathlib import Path

import diag11_core as core
import conf14_evaluation as historical
import lim17_protocol as protocol
import lim17_evaluation as evaluation
import lim17_reports as reports
import lim17_io as io17

LIMITS = [
    'Calibração e verificação são partes das validações já usadas no desenvolvimento; não constituem teste final independente.',
    'Ajuste restrito aos limiares de cadeira e mesa, uma grade de 49 opções e uma divisão agrupada; não garante melhora fora desses dados.',
    'As referências e os rótulos são preservados; diferenças semânticas existentes na avaliação permanecem.',
    'Propostas pós-NMS em confiança 0,001 e max_det=300; não há acesso a logits nem a propostas descartadas antes do cache.',
    'Alertas medem concordância com a política V2; não validam risco físico, distância ou uso assistivo real.',
    'Origem do treino original do baseline permanece não determinada.',
    'Nenhum modelo ou configuração em uso é substituído automaticamente.']

def save_gz(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f, gzip.GzipFile(fileobj=f, mode='wb', mtime=0, compresslevel=4) as z:
        z.write(protocol.canonical(value).encode())
    tmp.replace(path)

def _subset(result, ids, partition):
    images = [{**x, 'partition': partition} for x in result['images'] if x['id'] in ids]
    events = [x for x in result['events'] if x['id'] in ids]
    summary, classes = historical.summarise(images, events, .25)
    summary.update({'modelo': result['summary']['modelo'], 'partition': partition})
    return {'summary': summary, 'classes': classes, 'images': images, 'events': events}

def _save_result(path, result):
    core.save_json(path / 'metricas.json', result['summary'])
    core.save_csv(path / 'metricas_por_classe.csv', result['classes'])
    core.save_csv(path / 'alertas_por_imagem.csv', result['images'])
    core.save_csv(path / 'avaliacao_objetos.csv', result['events'])

def _same_counts(a, b):
    fields = ['tp', 'fn', 'fp']
    return (all(a['alertas_por_objeto'][f] == b['alertas_por_objeto'][f] for f in fields)
            and a['alta_sem_alerta'] == b['alta_sem_alerta'] and a['alta_ref'] == b['alta_ref']
            and all(a['detector_no_limiar'][f] == b['detector_no_limiar'][f] for f in fields)
            and a['gt_ignorados_sem_cobertura'] == b['gt_ignorados_sem_cobertura'])

def run(cohorts, caches, source, partitions, split_rows, split_audit, policy, matcher,
        output, context_key, backup=lambda stage: None):
    """A verificação recebe uma única configuração, nunca a grade de candidatos."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if set(caches) != {'baseline', 'candidato_10'}:
        raise ValueError('Somente baseline e candidato 10 congelados pertencem ao protocolo.')
    for partition in protocol.PARTITIONS:
        for cohort in protocol.COHORTS:
            if any(r['partition'] != partition for r in partitions[partition][cohort]):
                raise ValueError('Partição declarada de forma inconsistente.')
    if protocol.digest(split_rows) != split_audit['split_sha256']:
        raise RuntimeError('A divisão mudou após ser registrada.')
    for cohort in protocol.COHORTS:
        expected_ids = {r['id'] for r in cohorts[cohort]}
        sets = [{r['id'] for r in partitions[p][cohort]} for p in protocol.PARTITIONS]
        if sets[0] & sets[1] or sets[0] | sets[1] != expected_ids:
            raise ValueError('Calibração e verificação precisam ser disjuntas e cobrir a coorte.')

    refs, reproduction, image_checks, gt_checks = {}, [], [], []
    for model in ['baseline', 'candidato_10']:
        refs[model] = {}
        for cohort in protocol.COHORTS:
            result = evaluation.evaluate(cohorts[cohort], caches[model][cohort], protocol.profile(),
                                         policy, matcher, model, output / 'referencias_025' / model / cohort)
            refs[model][cohort] = result
            status, images, objects = evaluation.compare_reference(result, source['references'][model][cohort], cohort, model)
            reproduction.append(status)
            image_checks.extend(images)
            gt_checks.extend(objects)
    core.save_csv(output / 'reproducao_025_por_imagem.csv', image_checks)
    core.save_csv(output / 'reproducao_025_por_objeto.csv', gt_checks)
    core.save_json(output / 'reproducao_025_resumo.json', reproduction)
    if not all(r['reproduziu'] for r in reproduction):
        raise RuntimeError('A referência em 0,25 não foi reproduzida. A busca não será iniciada.')

    rules = {'completa_descritiva': source['rule']}
    subset_refs = {}
    comparisons = []
    for partition in protocol.PARTITIONS:
        subset_refs[partition] = {m: {} for m in refs}
        for model in refs:
            for cohort in protocol.COHORTS:
                ids = {r['id'] for r in partitions[partition][cohort]}
                result = _subset(refs[model][cohort], ids, partition)
                subset_refs[partition][model][cohort] = result
                comparisons.append(reports.comparison_row(partition, model, cohort, result['summary']))
        rules[partition] = protocol.rule_from_references(
            subset_refs[partition]['baseline'][protocol.COHORTS[0]]['summary'],
            subset_refs[partition]['candidato_10'][protocol.COHORTS[1]]['summary'])
    for model in refs:
        for cohort in protocol.COHORTS:
            comparisons.append(reports.comparison_row('completa_descritiva', model, cohort, refs[model][cohort]['summary']))
    protocol.immutable_json(output / 'regras_por_particao.json', rules)
    backup('referências reproduzidas e regras congeladas')

    books, frontier_checks = {}, []
    for cohort in protocol.COHORTS:
        records = partitions['calibracao'][cohort]
        path = output / 'fronteiras_calibracao' / (cohort + '.json.gz')
        marker = path.with_suffix('.concluido.json')
        request = {'context_key': context_key, 'ids': [r['id'] for r in records],
                   'partition': 'calibracao', 'grid': list(protocol.GRID)}
        meta = json.loads(marker.read_text()) if marker.is_file() else {}
        if path.is_file() and meta.get('request') == request and meta.get('sha256') == core.sha256_file(path):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                books[cohort] = json.load(f)
            print('Fronteira de calibração recuperada:', cohort, flush=True)
        else:
            books[cohort] = evaluation.calibration_frontier(records, caches['candidato_10'][cohort], policy, matcher,
                                                           progress=lambda message: print(message, flush=True))
            save_gz(path, books[cohort])
            core.save_json(marker, {'request': request, 'sha256': core.sha256_file(path)})
        expected = subset_refs['calibracao']['candidato_10'][cohort]['summary']
        actual = evaluation.frontier_summary(books[cohort], protocol.profile())
        ok = _same_counts(expected, actual)
        frontier_checks.append({'coorte': cohort, 'perfil': 'uniforme_025', 'contagens_iguais': ok})
        if not ok:
            raise RuntimeError('O cálculo por classe não reproduziu a avaliação integral em calibração.')
        backup('fronteira calculada: ' + cohort)
    core.save_json(output / 'auditoria_fronteira_025.json', frontier_checks)
    grid = evaluation.search_grid(books, rules['calibracao'])
    selection = protocol.choose(grid)
    choice_record = {'context_key': context_key, 'split_sha256': split_audit['split_sha256'],
                     'grid_sha256': protocol.digest(grid), 'selection': selection,
                     'verification_configurations_allowed': 1 if selection['selected'] else 0,
                     'verification_result_used_for_selection': False}
    choice_sha = protocol.immutable_json(output / 'escolha_congelada_antes_verificacao.json', choice_record)
    core.save_csv(output / 'grade_calibracao.csv', reports.grid_rows(grid))
    # O backup precisa terminar antes de qualquer avaliação com limiar novo na verificação.
    backup('escolha congelada ANTES da verificação')
    if protocol.digest(json.loads((output / 'escolha_congelada_antes_verificacao.json').read_text())) != choice_sha:
        raise RuntimeError('Escolha alterada depois do congelamento.')

    verification = full_checks = None
    selected = selection['selected']
    status = 'nenhuma_configuracao_elegivel_na_calibracao'
    message = 'Nenhuma das 49 configurações cumpriu os requisitos na calibração. Nenhuma configuração nova foi avaliada na verificação.'
    if selected:
        thresholds = protocol.validate_profile(selected['thresholds'])
        selected_results = {p: {} for p in protocol.PARTITIONS}
        for cohort in protocol.COHORTS:
            result = evaluation.evaluate(partitions['calibracao'][cohort], caches['candidato_10'][cohort], thresholds,
                                         policy, matcher, selected['id'], output / 'escolha' / 'calibracao' / cohort)
            expected = selected['home' if cohort == protocol.COHORTS[0] else 'external']
            if not _same_counts(expected, result['summary']):
                raise RuntimeError('A escolha recalculada integralmente diverge da grade; verificação bloqueada.')
            selected_results['calibracao'][cohort] = result
        for cohort in protocol.COHORTS:
            result = evaluation.evaluate(partitions['verificacao_interna'][cohort], caches['candidato_10'][cohort], thresholds,
                                         policy, matcher, selected['id'], output / 'escolha' / 'verificacao_interna' / cohort)
            selected_results['verificacao_interna'][cohort] = result
        verification = protocol.judge(selected_results['verificacao_interna'][protocol.COHORTS[0]]['summary'],
                                      selected_results['verificacao_interna'][protocol.COHORTS[1]]['summary'], rules['verificacao_interna'])
        protocol.immutable_json(output / 'verificacao_unica.json', {
            'context_key': context_key, 'choice_sha256': choice_sha, 'profile_id': selected['id'],
            'thresholds': thresholds, 'distinct_configurations_evaluated': 1, 'second_choice_attempts': 0,
            'checks': verification})
        backup('verificação única concluída; sem nova seleção')
        full = {}
        for partition in protocol.PARTITIONS:
            for cohort in protocol.COHORTS:
                comparisons.append(reports.comparison_row(partition, selected['id'], cohort, selected_results[partition][cohort]['summary']))
        for cohort in protocol.COHORTS:
            im = [{**x} for p in protocol.PARTITIONS for x in selected_results[p][cohort]['images']]
            events = [x for p in protocol.PARTITIONS for x in selected_results[p][cohort]['events']]
            summary, classes = historical.summarise(im, events, .25)
            summary.update({'modelo': selected['id'], 'partition': 'completa_descritiva',
                            'conf_alertas': .25 if thresholds == protocol.profile() else None,
                            'conf_alertas_por_classe': thresholds, 'ap_recalculada': False})
            for row in classes:
                row['conf_alertas'] = thresholds[row['classe_id']]
            for row in im:
                row['partition'] = 'completa_descritiva'
            full[cohort] = {'summary': summary, 'classes': classes, 'images': im, 'events': events}
            _save_result(output / 'escolha' / 'completa_descritiva' / cohort, full[cohort])
            comparisons.append(reports.comparison_row('completa_descritiva', selected['id'], cohort, summary))
        full_checks = protocol.judge(full[protocol.COHORTS[0]]['summary'], full[protocol.COHORTS[1]]['summary'], source['rule'])
        if not verification['elegivel']:
            status = 'configuracao_nao_confirmada_na_verificacao'
            message = 'A configuração escolhida em calibração falhou na verificação interna. Nenhuma segunda configuração foi tentada; o baseline permanece como referência.'
        elif not full_checks['elegivel']:
            status = 'configuracao_nao_cumpre_limites_globais'
            message = 'A configuração passou na comparação local de verificação, mas não cumpriu todos os seis limites numéricos históricos na validação completa. Nenhum modelo foi substituído.'
        else:
            status = 'configuracao_promissora_em_verificacao_interna'
            message = 'A configuração passou na verificação interna e nos seis limites numéricos históricos. É candidata para avaliação final independente; nenhum modelo foi substituído.'
        core.save_json(output / 'configuracao_inferencia_candidata.json', {
            'model': 'candidato_10', 'thresholds': thresholds, 'profile_id': selected['id'],
            'context_key': context_key, 'aprovada_no_protocolo_interno': verification['elegivel'] and full_checks['elegivel'],
            'implantada': False, 'treinamento_executado': False, 'uso': 'filtrar propostas pós-NMS e recalcular correspondências e política V2'})
    summary = {'version': protocol.VERSION, 'concluido': True, 'status': status, 'mensagem': message,
               'context_key': context_key, 'primary_endpoint': 'uma_escolha_calibracao_vs_referencias_na_verificacao_interna',
               'configurations_evaluated_calibration': 49, 'eligible_calibration': selection['eligible_count'],
               'distinct_configurations_evaluated_verification': 1 if selected else 0,
               'selected_profile_id': selected['id'] if selected else None,
               'selected_thresholds': selected['thresholds'] if selected else None,
               'avaliacao_escolha': {'verificacao_interna': verification, 'limites_globais_descritivos': full_checks},
               'split': split_audit, 'training_executed': False, 'weights_changed': False,
               'labels_changed': False, 'model_replaced': False, 'test_home_inferred': False,
               'test_oi20k_inferred': False, 'future_reserve_extracted': False, 'future_reserve_inferred': False,
               'verification_is_independent_final_test': False, 'manual_form_required': False,
               'source16_protocol_changed': False, 'source16_reproduction_passed': all(x['reproduziu'] for x in reproduction),
               'limites': LIMITS}
    reports.write_report(output, summary, grid, rules, selection, comparisons)
    core.save_json(output / 'integridade_final_17.json', {
        'context_key': context_key, 'choice_sha256': choice_sha, 'split_sha256': split_audit['split_sha256'],
        'grade_usou_somente_calibracao': True, 'verificacao_usou_no_maximo_uma_configuracao': True,
        'reproducao_025_conferida': summary['source16_reproduction_passed'],
        'rotulos_e_pesos_preservados': True, 'testes_e_reserva_inferidos': False})
    (output / 'execucao_interrompida.json').unlink(missing_ok=True)
    package = io17.package(output)
    backup('resultado e pacote final')
    return summary, package
