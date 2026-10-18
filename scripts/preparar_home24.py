"""Empacota a seleção Home 24 usando os bytes originais e seu manifesto fixado."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'datasets/manifestos/home24.json.gz'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, output, manifest, verify_only=False):
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError('ZIP fonte não encontrado: ' + str(source))
    if not zipfile.is_zipfile(source):
        raise ValueError('A origem precisa ser um ZIP válido.')
    actual = sha_file(source)
    if actual != manifest['home_zip_sha256']:
        raise ValueError('O ZIP não corresponde à fonte histórica Home/SUN fixada. '
                         'O download público atual não substitui automaticamente a preparação histórica.')
    expected = [('train', manifest['train']), ('val', manifest['val']),
                ('test', manifest['test_descriptors'])]
    ids = [r['id'] for _, rows in expected for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Manifesto contém IDs repetidos entre as partições.')
    report = {'source_name': source.name, 'source_sha256': actual,
              'dataset_id': manifest['dataset_id'],
              'images': {split: len(rows) for split, rows in expected},
              'images_and_labels_checked': 0, 'test_historical': True,
              'new_split_created': False, 'labels_changed': False,
              'training_executed': False, 'portable_archive': not verify_only}
    if not verify_only and output.exists():
        raise FileExistsError('A saída já existe; escolha outro caminho: ' + str(output))
    temp_path = None
    archive = None
    try:
        if not verify_only:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=output.name + '.', suffix='.part',
                                              dir=output.parent, delete=False) as f:
                temp_path = Path(f.name)
            archive = zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_DEFLATED,
                                      compresslevel=6, allowZip64=True)
        with zipfile.ZipFile(source) as original:
            names = original.namelist()
            if len(names) != len(set(names)):
                raise ValueError('ZIP fonte contém nomes de entrada repetidos.')
            destinations = set()
            for split, rows in expected:
                for record in rows:
                    token = hashlib.sha256(record['id'].encode()).hexdigest()[:24]
                    for key, hash_key, directory in [('member', 'sha256', 'images'),
                                                       ('label_member', 'label_sha256', 'labels')]:
                        member = record[key]
                        p = PurePosixPath(member)
                        if p.is_absolute() or '..' in p.parts or '\\' in member or ':' in member:
                            raise ValueError('Caminho de membro inválido no manifesto.')
                        raw = original.read(member)
                        if hashlib.sha256(raw).hexdigest() != record[hash_key]:
                            raise ValueError('Imagem ou rótulo difere do manifesto: ' + member)
                        suffix = p.suffix if directory == 'images' else '.txt'
                        dest = f'{directory}/{split}/{token}{suffix}'
                        if dest in destinations:
                            raise ValueError('Colisão de nomes de saída: ' + dest)
                        destinations.add(dest)
                        if archive:
                            archive.writestr(dest, raw)
                        report['images_and_labels_checked'] += 1
        if archive:
            yaml = 'path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 12\nnames:\n'
            yaml += ''.join(f'  {i}: {json.dumps(name, ensure_ascii=False)}\n'
                            for i, name in enumerate(manifest['classes']))
            archive.writestr('data.yaml', yaml)
            archive.writestr('manifesto_historico.json', json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr('verificacao_pacote.json', json.dumps(report, ensure_ascii=False, indent=2))
            archive.writestr('LEIA_ME.md', '# Home 24 — pacote para reprodução\n\n'
                             'Mesmas imagens, rótulos e partições do manifesto arquivado. '
                             'O TEST é histórico e previamente consultado. '
                             'Esta embalagem não substitui o ZIP fonte nos notebooks que verificam seu hash.\n\n'
                             'Fonte: HomeObjects-3K, Ultralytics. Atribuição e licença indicada AGPL-3.0: '
                             'https://docs.ultralytics.com/datasets/detect/homeobjects-3k\n'
                             'Alteração: seleção e organização de partições; bytes de imagens/rótulos preservados.\n')
            archive.close()
            archive = None
            temp_path.replace(output)
            temp_path = None
            report['output_name'] = output.name
            report['output_bytes'] = output.stat().st_size
            report['output_sha256'] = sha_file(output)
            report_path = output.with_suffix('.verificacao.json')
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return report
    finally:
        if archive:
            archive.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origem', required=True, type=Path,
                        help='Caminho do homeobjects_sunrgbd_validado.zip histórico.')
    parser.add_argument('--saida', type=Path,
                        default=ROOT / 'publicacao/assets/base_home24_reproducao.zip')
    parser.add_argument('--somente-conferir', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(gzip.decompress(MANIFEST.read_bytes()))
    if {k: len(manifest[v]) for k, v in [('train', 'train'), ('val', 'val'),
                                       ('test', 'test_descriptors')]} != {'train': 1818, 'val': 456, 'test': 403}:
        parser.error('O manifesto distribuído não corresponde ao Home 24.')
    try:
        report = prepare(args.origem, args.saida, manifest, args.somente_conferir)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        parser.exit(2, 'Falha: ' + str(e) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
