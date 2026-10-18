"""Autentica o pacote 16 e reutiliza caches completos, com inferência só se faltar."""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import diag11_core as core
import conf14_evaluation as historical
import lim17_protocol as protocol
from lim17_io import DigestMismatch

MODELS = ('baseline', 'candidato_10')
PREDICT_ARGS = {'conf': .001, 'imgsz': 640, 'rect': True, 'iou': .70,
                'max_det': 300, 'half': False, 'batch': 1}

def rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))

def load_source16(path, expected, output=None):
    if core.sha256_file(path) != expected['pacote_16']:
        raise RuntimeError('O pacote 16 difere do resultado recebido.')
    result = {'references': {}, 'cache_manifests': {}}
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())) or z.testzip() is not None:
            raise RuntimeError('Pacote 16 com membros repetidos ou CRC inválido.')
        for name in z.namelist():
            core.safe_member(name)
        summary = json.loads(z.read('resumo_experimento_16.json'))
        cfg = json.loads(z.read('configuracao_experimento_16.json'))
        if (not summary['concluido'] or summary['reserva_futura_inferida'] or summary['test_home_inferido']
                or summary['test_oi20k_inferido'] or cfg['context_key'] != expected['contexto_16']):
            raise RuntimeError('Resultado 16 incompatível ou incompleto.')
        if cfg['plano']['initial_checkpoint_sha256'] != expected['candidato']:
            raise RuntimeError('Candidato inicial diferente do esperado.')
        for name, sha in cfg['plano']['codigo_sha256'].items():
            if hashlib.sha256(z.read('codigo/' + name)).hexdigest() != sha:
                raise RuntimeError('Código alterado no pacote 16: ' + name)
        rule_raw = z.read('regra_selecao_congelada_12.json')
        if hashlib.sha256(rule_raw).hexdigest() != expected['regra_12_sha256']:
            raise RuntimeError('Regra histórica alterada.')
        result['rule_raw'], result['rule'] = rule_raw, json.loads(rule_raw)
        result['source_folder'] = 'resultados/' + cfg['experimento']
        if result['source_folder'] != expected['pasta_16']:
            raise RuntimeError('Pasta do resultado 16 incompatível.')
        for model in MODELS:
            result['references'][model], result['cache_manifests'][model] = {}, {}
            for cohort in protocol.COHORTS:
                prefix = 'referencias/' + model + '/' + cohort + '/'
                result['references'][model][cohort] = {
                    'summary': json.loads(z.read(prefix + 'metricas.json')),
                    'images': rows(z.read(prefix + 'alertas_por_imagem.csv')),
                    'events': rows(z.read(prefix + 'avaliacao_objetos.csv'))}
                meta = json.loads(z.read(prefix + 'cache/cache_manifesto.json'))
                key = 'baseline' if model == 'baseline' else 'candidato'
                request = meta['solicitacao']
                if (not meta['concluido'] or request['peso_sha256'] != expected[key]
                        or request['parametros'] != PREDICT_ARGS or request['coorte'] != cohort):
                    raise RuntimeError('Manifesto do cache 16 incompatível.')
                result['cache_manifests'][model][cohort] = meta
        if output:
            target = Path(output)
            target.mkdir(parents=True, exist_ok=True)
            (target / 'regra_selecao_congelada_12.json').write_bytes(rule_raw)
            core.save_json(target / 'fonte_autenticada_16.json', {
                'pacote_sha256': expected['pacote_16'], 'contexto_16': cfg['context_key'],
                'pasta_16': result['source_folder'], 'status_16': summary['status']})
    return result

def signature(records):
    return [{k: r[k] for k in ['id', 'sha256', 'label_sha256', 'width', 'height', 'known_mask', 'role']} for r in records]

def validate_source_records(cohorts, source):
    if set(cohorts) != set(protocol.COHORTS):
        raise ValueError('Coortes diferentes do protocolo.')
    for cohort, records in cohorts.items():
        if len(records) != (456 if cohort == protocol.COHORTS[0] else 1000):
            raise ValueError('Quantidade de imagens mudou: ' + cohort)
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('IDs repetidos.')
        for model in MODELS:
            request = source['cache_manifests'][model][cohort]['solicitacao']
            actual = hashlib.sha256(core.canonical(signature(records)).encode()).hexdigest()
            if actual != request['entradas_sha256']:
                raise RuntimeError('A ordem, os bytes, as dimensões ou a cobertura diferem do cache 16: ' + cohort)
            if set(r['id'] for r in records) != {r['id'] for r in source['references'][model][cohort]['images']}:
                raise RuntimeError('IDs diferentes da referência 16.')
        expected_role = 'home_val' if cohort == protocol.COHORTS[0] else 'val_nova'
        if any(r['role'] != expected_role for r in records):
            raise RuntimeError('Papel proibido na validação.')
    return True

def infer_frozen(weight, records, directory, expected_weight, predict_fn=None, model_factory=None):
    """Sem model.train/fit: uma passagem de inferência no peso autenticado."""
    if not records or any(r['role'] not in {'home_val', 'val_nova'} for r in records):
        raise ValueError('Inferência em dado protegido ou vazio bloqueada.')
    if core.sha256_file(weight) != expected_weight:
        raise RuntimeError('Peso de inferência diferente da referência.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    environment = {'inferencia': 'callback_de_teste'}
    if predict_fn is None:
        import importlib.metadata
        import subprocess
        import sys
        try:
            version = importlib.metadata.version('ultralytics')
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != '8.4.150':
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.150'], check=True)
        import torch
        import ultralytics
        import cv2
        if ultralytics.__version__ != '8.4.150':
            raise RuntimeError('A inferência de recuperação exige ultralytics==8.4.150. Reinicie a sessão após a instalação.')
        from ultralytics import YOLO, settings
        settings.update({k: False for k in ['sync', 'hub', 'platform', 'wandb', 'comet', 'mlflow', 'neptune', 'clearml', 'dvc', 'tensorboard'] if k in settings})
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cv2.setNumThreads(0)
        device = 0 if torch.cuda.is_available() else 'cpu'
        model = (model_factory or YOLO)(str(weight))
        if [str(model.names[i]) for i in range(12)] != core.NAMES:
            raise RuntimeError('Classes do peso incompatíveis.')
        for p in model.model.parameters():
            p.requires_grad_(False)
        model.model.eval()
        predict_fn = lambda r: core.model_predictions(model, r, device)
        environment = {'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
                       'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device == 0 else None}
    path = directory / 'predicoes_conf_0p001.jsonl.gz'
    tmp = path.with_suffix('.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=4) as f:
        for i, r in enumerate(records, 1):
            if r['role'] not in {'home_val', 'val_nova'}:
                raise ValueError('Inferência em dado protegido bloqueada.')
            preds = predict_fn(r)
            historical.validate_predictions(preds, r)
            f.write(json.dumps({'id': r['id'], 'image_sha256': r['sha256'],
                                'label_sha256': r['label_sha256'], 'predicoes': preds}, ensure_ascii=False) + '\n')
            if i % 100 == 0 or i == len(records):
                print(f'Inferência de recuperação {r["coorte"]}: {i}/{len(records)}', flush=True)
    tmp.replace(path)
    meta = {'source': 'inferencia_17_sem_treino', 'peso_sha256': expected_weight,
            'entradas_sha256': hashlib.sha256(core.canonical(signature(records)).encode()).hexdigest(),
            'parametros': PREDICT_ARGS, 'ambiente_inferencia': environment,
            'cache_sha256': core.sha256_file(path), 'concluido': True}
    core.save_json(directory / 'cache_manifesto_17.json', meta)
    return path, meta

def obtain_cache(model, cohort, records, source, expected, drive, work, output, infer_fn=infer_frozen):
    expected_role = 'home_val' if cohort == protocol.COHORTS[0] else 'val_nova'
    if model not in MODELS or cohort not in protocol.COHORTS or not records or any(r['role'] != expected_role for r in records):
        raise ValueError('Modelo, papel ou coorte proibidos no cache.')
    work, output = Path(work), Path(output)
    original = source['cache_manifests'][model][cohort]
    relative = source['source_folder'] + '/referencias/' + model + '/' + cohort + '/cache/predicoes_conf_0p001.jsonl.gz'
    local = work / 'caches_fontes' / model / cohort / 'predicoes_conf_0p001.jsonl.gz'
    target = output / 'caches_recriados' / model / cohort
    try:
        path = drive.fetch(relative, local, original['cache_sha256'], optional=True)
    except DigestMismatch as exc:
        core.save_json(output / 'auditoria_cache' / model / (cohort + '_cache_fonte_rejeitado.json'), {'motivo': str(exc), 'acao': 'recriar por inferencia'})
        path = None
    if path is not None:
        meta = {'source': 'cache_autenticado_16', 'cache_sha256': original['cache_sha256'],
                'source_request': original['solicitacao'], 'ambiente_atual_igual_ao_da_captura_exigido': False,
                'motivo': 'propostas congeladas por SHA; não ocorre inferência ao reutilizar'}
    else:
        key = 'baseline' if model == 'baseline' else 'candidato'
        prior_path, prior_meta = target / 'predicoes_conf_0p001.jsonl.gz', target / 'cache_manifesto_17.json'
        prior = json.loads(prior_meta.read_text()) if prior_meta.is_file() else {}
        request_sha = hashlib.sha256(core.canonical(signature(records)).encode()).hexdigest()
        if (prior_path.is_file() and prior.get('cache_sha256') == core.sha256_file(prior_path)
                and prior.get('peso_sha256') == expected[key] and prior.get('entradas_sha256') == request_sha
                and prior.get('parametros') == PREDICT_ARGS and prior.get('concluido')):
            path, meta = prior_path, prior
        else:
            relative_weight = ('modelos/yolov8n_baseline/best.pt' if model == 'baseline'
                               else expected['pasta_10'] + '/best.pt')
            weight = drive.fetch(relative_weight, work / 'pesos' / (model + '.pt'), expected[key])
            path, meta = infer_fn(weight, records, target, expected[key])
        meta = {**meta, 'cache_16_original_sha256': original['cache_sha256']}
    if not path.is_file() or core.sha256_file(path) != meta['cache_sha256']:
        raise RuntimeError('Cache indisponível ou alterado após recuperação.')
    cached = historical.read_cache(path, records)
    core.save_json(output / 'auditoria_cache' / model / (cohort + '.json'), meta)
    print('Cache pronto:', model, cohort, meta['source'], flush=True)
    return cached, meta
