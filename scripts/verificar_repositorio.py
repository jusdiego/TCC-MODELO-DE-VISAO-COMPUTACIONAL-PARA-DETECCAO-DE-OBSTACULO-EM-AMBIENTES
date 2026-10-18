"""Confere integridade, partições e métricas arquivadas, sem executar treinamento."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(relative):
    p = ROOT / relative
    raw = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
    return json.loads(raw)


def rows(relative):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), label)


def verify_profile(profile):
    counts = profile['detector']
    tp, fn, fp = [counts[k] for k in ['tp', 'fn', 'fp']]
    close(counts['precision'], tp / (tp + fp), 'Precisão não corresponde às contagens.')
    close(counts['recall'], tp / (tp + fn), 'Recall não corresponde às contagens.')
    close(counts['f1'], 2 * tp / (2 * tp + fn + fp), 'F1 não corresponde às contagens.')
    for group in ['classes', 'image_rows']:
        require(len(profile['image_rows']) == profile['images'], 'Quantidade de imagens divergente.')
        for key in ['tp', 'fn', 'fp']:
            require(sum(r[key] for r in profile[group]) == counts[key],
                    f'Contagens de {group} não fecham: {key}.')


def verify():
    checksum_file = ROOT / 'SHA256SUMS'
    require(checksum_file.is_file(), 'SHA256SUMS não encontrado.')
    checksum_count = 0
    for line in checksum_file.read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = ROOT / relative
        require(path.is_file(), 'Arquivo ausente: ' + relative)
        require(sha(path) == expected, 'Hash divergente: ' + relative)
        checksum_count += 1
    origin_rows = rows('auditoria/catalogo_origens.csv')
    for row in origin_rows:
        path = ROOT / row['arquivo']
        require(sha(path) == row['sha256'], 'Cópia histórica alterada: ' + row['arquivo'])
        if row.get('sha256_descompactado'):
            require(hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
                    == row['sha256_descompactado'], 'Manifesto descompactado alterado.')
    index = rows('docs/indice_notebooks.csv')
    require(len(index) == 71, 'Índice deve conter 71 notebooks.')
    require(sum(r['grupo'] == 'principal' for r in index) == 32, 'Quantidade principal divergente.')
    outputs = 0
    for row in index:
        path = ROOT / row['arquivo']
        require(sha(path) == row['sha256_organizado'], 'Notebook difere da cópia organizada.')
        notebook = json.loads(path.read_text(encoding='utf-8'))
        require(notebook['nbformat'] == 4, 'Formato de notebook inesperado.')
        outputs += sum(len(c.get('outputs', [])) for c in notebook['cells'])
    home = load('datasets/manifestos/home24.json.gz')
    home_parts = [home['train'], home['val'], home['test_descriptors']]
    require([len(r) for r in home_parts] == [1818, 456, 403], 'Partições Home divergentes.')
    require(len({r['id'] for part in home_parts for r in part}) == 2677, 'IDs Home repetidos.')
    for split, part in zip(['train', 'val', 'test'], home_parts):
        require({r['id'] for r in rows(f'datasets/divisoes/home24_{split}.csv')}
                == {r['id'] for r in part}, 'Lista Home difere do manifesto.')
    nyu = load('datasets/manifestos/nyu22_desenvolvimento.json.gz')
    require([len(nyu['train']), len(nyu['val']), len(nyu['test_ids'])] == [634, 161, 654],
            'Partições NYU divergentes.')
    scenes = [set(r['scene_group'] for r in nyu[k]) for k in ['train', 'val']]
    require(not scenes[0] & scenes[1], 'Sobreposição de cenas NYU treino/VAL.')
    require(not (scenes[0] | scenes[1]) & set(nyu['test_scenes']), 'Sobreposição de cenas com NYU TEST.')
    oi = load('datasets/manifestos/divisao_openimages.json.gz')
    require(len(oi['ids_treino']) == 80000 and len(oi['ids_teste']) == 20000, 'Divisão Open Images divergente.')
    require(not set(oi['ids_treino']) & set(oi['ids_teste']), 'IDs Open Images repetidos entre partições.')
    final = load('resultados/experimentos/24/resumo_experimento_24.json')
    require(final['completed'] and final['test_is_new_independent'] is False, 'Estado científico alterado.')
    for key in ['validation', 'validation_secondary', 'test_historico', 'test_historico_secondary']:
        for profile in final[key].values():
            verify_profile(profile)
    verify_profile(final['external_nyu']['metrics'])
    comparison = rows('resultados/experimentos/24/comparacao_24.csv')
    for row in comparison:
        key = 'validation' if row['conjunto'] == 'Home VAL' else 'test_historico'
        profile = final[key][row['modelo']]
        for metric in ['precision', 'recall', 'f1', 'tp', 'fn', 'fp']:
            close(row[metric], profile['detector'][metric], 'Tabela de comparação não fecha.')
    for row in rows('resultados/experimentos/24/classes_test_historico_24.csv'):
        cls = final['test_historico'][row['modelo']]['classes'][int(row['class_id'])]
        for metric in ['precision', 'recall', 'f1', 'tp', 'fn', 'fp', 'gt']:
            close(row[metric], cls[metric], 'Tabela por classe não fecha.')
    require(final['run']['epochs'] == 25 and final['best']['epoch'] == 13, 'Histórico final RF divergente.')
    rf = final['test_historico']['rfdetr_small24']['detector']
    return {'status': 'ok', 'arquivos_conferidos': checksum_count,
            'copias_historicas_conferidas': len(origin_rows), 'notebooks': len(index),
            'principais': 32, 'variantes': 39, 'saidas_preservadas': outputs,
            'home': [1818, 456, 403], 'nyu': [634, 161, 654],
            'openimages': [80000, 20000], 'rf_home': rf,
            'novos_treinamentos': 0, 'novas_inferencias': 0}


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as e:
        print('Falha na conferência: ' + str(e), file=sys.stderr)
        raise SystemExit(1)
