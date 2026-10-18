"""Localiza e confere os ZIPs de dados históricos em uma pasta fornecida."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(directory, expected):
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError('Pasta de origem não encontrada: ' + str(directory))
    names = {r['nome'] for r in expected}
    found = defaultdict(list)
    for p in directory.rglob('*.zip'):
        if p.is_file() and not p.is_symlink() and p.name in names:
            found[p.name].append(p)
    hashes = {}
    report = []
    for item in expected:
        candidates = []
        for p in sorted(found[item['nome']]):
            if p not in hashes:
                hashes[p] = sha_file(p)
            candidates.append({'arquivo': str(p), 'bytes': p.stat().st_size,
                               'sha256': hashes[p], 'confere': hashes[p] == item['sha256']})
        status = 'ok' if any(c['confere'] for c in candidates) else ('divergente' if candidates else 'ausente')
        report.append({'grupo': item['grupo'], 'nome': item['nome'],
                       'sha256_esperado': item['sha256'], 'status': status, 'candidatos': candidates})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origem', type=Path, required=True)
    parser.add_argument('--saida', type=Path, default=ROOT / 'publicacao/assets/verificacao_bases.json')
    args = parser.parse_args()
    expected = json.loads((ROOT / 'datasets/arquivos_esperados.json').read_text(encoding='utf-8'))['arquivos']
    try:
        rows = check(args.origem, expected)
        summary = dict(Counter(r['status'] for r in rows))
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(json.dumps({'resumo': summary, 'arquivos': rows},
                                        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except OSError as e:
        parser.exit(2, 'Falha: ' + str(e) + '\n')
    print(json.dumps({'resumo': summary, 'relatorio': str(args.saida)}, ensure_ascii=False, indent=2))
    return 0 if all(r['status'] == 'ok' for r in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
