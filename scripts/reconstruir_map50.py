"""Reconstrói o mAP50 do Colab com previsões arquivadas, sem treino ou inferência.

Aceita o pacote_experimento_24.zip original ou o ZIP público de apresentação.
Os GTs e seus hashes vêm dos registros históricos já versionados no repositório.
Dependências: Python >= 3.10 e numpy >= 1.24.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / (
    "resultados/iniciais/pacotes/avaliacao_final_test-20261015T135436Z-1-001/"
    "avaliacao_final_test"
)
NAMES = [
    "bed", "sofa", "chair", "table", "lamp", "tv", "laptop", "wardrobe",
    "window", "door", "potted plant", "photo frame",
]
MODELS = ["candidato10", "21_epoca21", "rfdetr_small24"]
# Referências da conferência anterior; usadas somente depois do cálculo.
EXPECTED = {
    "candidato10": 0.6748580838720697,
    "21_epoca21": 0.6787673977160485,
    "rfdetr_small24": 0.7899370204212569,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_json(data):
    raw = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_gt(csv_path):
    images = {}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not row["bbox_ref"] or not row["classe_ref"]:
                continue
            name = row["imagem"]
            index = int(float(row["id_gt"]))
            obj = {
                "classe_id": NAMES.index(row["classe_ref"]),
                "bbox": json.loads(row["bbox_ref"]),
            }
            b = obj["bbox"]
            require(
                len(b) == 4 and all(math.isfinite(v) for v in b)
                and b[2] > b[0] and b[3] > b[1],
                "GT inválido: " + name,
            )
            part = images.setdefault(name, {})
            require(index not in part or part[index] == obj, "GT conflitante: " + name)
            part[index] = obj
    # A ordem de id_gt preserva os índices de anotação do avaliador pontual.
    result = {name: [rows[i] for i in sorted(rows)] for name, rows in images.items()}
    require(len(result) == 403, "O TEST precisa conter 403 imagens.")
    require(sum(map(len, result.values())) == 3465, "Esperados 3.465 objetos.")
    return result


def iou_matrix(gt, predictions):
    if not gt or not predictions:
        return np.zeros((len(gt), len(predictions)), dtype=float)
    g = np.asarray([v["bbox"] for v in gt], dtype=float)
    p = np.asarray([v["bbox"] for v in predictions], dtype=float)
    start = np.maximum(g[:, None, :2], p[None, :, :2])
    end = np.minimum(g[:, None, 2:], p[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, end - start), axis=2)
    union = (
        np.prod(np.maximum(0.0, g[:, 2:] - g[:, :2]), axis=1)[:, None]
        + np.prod(np.maximum(0.0, p[:, 2:] - p[:, :2]), axis=1)[None, :]
        - intersection
    )
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def point_counts(gt, predictions):
    """Mesmo pareamento por IoU usado pelo perfil operacional do Colab."""
    matrix = iou_matrix(gt, predictions)
    candidates = [
        (float(matrix[gi, pi]), gi, pi)
        for gi, g in enumerate(gt)
        for pi, p in enumerate(predictions)
        if g["classe_id"] == p["classe_id"] and matrix[gi, pi] >= 0.5
    ]
    candidates.sort(key=lambda r: (-r[0], r[1], r[2]))
    used_g, used_p = set(), set()
    for _, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
    return len(used_g), len(gt) - len(used_g), len(predictions) - len(used_p)


def ap50(gt_by_name, cache, confidence_floor):
    """AP em 101 recalls; pareamento por confiança e melhor IoU disponível.

    Cada anotação é usada no máximo uma vez. A precisão é interpolada pela
    envoltória não crescente, com zero além do recall máximo alcançado.
    Não adiciona NMS nem altera o limite de captura já aplicado pelos modelos.
    """
    result = {}
    for cid, name in enumerate(NAMES):
        n_gt = 0
        scores, flags = [], []
        for image_id, raw_predictions in cache.items():
            gt = [
                g for g in gt_by_name[image_id.rsplit("/", 1)[1]]
                if g["classe_id"] == cid
            ]
            n_gt += len(gt)
            predictions = sorted(
                (p for p in raw_predictions
                 if p["classe_id"] == cid and p["confianca"] >= confidence_floor),
                key=lambda p: -p["confianca"],
            )
            matrix = iou_matrix(gt, predictions)
            used = set()
            for pi, p in enumerate(predictions):
                candidates = [
                    (matrix[gi, pi], gi) for gi in range(len(gt))
                    if gi not in used and matrix[gi, pi] >= 0.5
                ]
                found = bool(candidates)
                if found:
                    _, gi = max(candidates)
                    used.add(gi)
                scores.append(p["confianca"])
                flags.append(found)
        require(n_gt > 0, "Classe sem GT: " + name)
        if not scores:
            result[name] = 0.0
            continue
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        correct = np.asarray(flags, dtype=bool)[order]
        tp, fp = np.cumsum(correct), np.cumsum(~correct)
        recall = tp / n_gt
        precision = tp / np.maximum(1, tp + fp)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        positions = np.searchsorted(recall, np.linspace(0.0, 1.0, 101), side="left")
        values = np.zeros(101)
        valid = positions < len(envelope)
        values[valid] = envelope[positions[valid]]
        result[name] = float(values.mean())
    return float(np.mean(list(result.values()))), result


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path)
        self.names = self.zip.namelist()
        require(len(set(self.names)) == len(self.names), "ZIP com membros repetidos.")
        candidates = [
            n for n in self.names if n.rsplit("/", 1)[-1] == "resumo_experimento_24.json"
        ]
        require(len(candidates) == 1, "Esperado um único resumo do 24 Colab no ZIP.")
        self.prefix = candidates[0][:-len("resumo_experimento_24.json")]

    def read_json(self, relative):
        return json.loads(self.zip.read(self.prefix + relative))

    def cache(self, model):
        base = self.prefix + f"test_historico/{model}/cache/"
        manifests = [
            n for n in self.names
            if n.startswith(base) and n.endswith("/manifesto_cache.json")
        ]
        require(len(manifests) == 1, "Cache ambíguo ou ausente: " + model)
        manifest = json.loads(self.zip.read(manifests[0]))
        request = manifest["request"]
        require(request["role"] == "test", "Cache não pertence ao teste.")
        require(request["capture_conf"] == 0.01, "Confiança de captura alterada.")
        require(request["max_det"] == 300, "Limite de captura alterado.")
        folder = manifests[0][:-len("manifesto_cache.json")]
        blocks = sorted(
            n for n in self.names
            if n.startswith(folder) and n.rsplit("/", 1)[-1].startswith("bloco_")
            and n.endswith(".json.gz")
        )
        data = {}
        for block in blocks:
            payload = json.loads(gzip.decompress(self.zip.read(block)))
            require(payload["key"] == manifest["key"], "Chave de bloco alterada.")
            require(digest_json(payload["data"]) == payload["data_sha256"],
                    "Hash do bloco divergente: " + block)
            require(not set(data) & set(payload["data"]), "ID repetido nos blocos.")
            data.update(payload["data"])
        require(list(data) == [r["id"] for r in request["input"]],
                "Blocos incompletos ou com ordem diferente do manifesto.")
        return data


def reconstruct(args):
    gt = load_gt(args.objetos_test)
    old_manifest = json.loads(Path(args.manifesto_test).read_text(encoding="utf-8"))
    source = Archive(args.pacote_colab)
    try:
        development = source.read_json("manifesto_desenvolvimento_24.json")
        descriptors = {r["name"]: r for r in development["test_descriptors"]}
        require(set(descriptors) == set(gt), "IDs de GT e manifesto do 24 diferem.")
        require(len(old_manifest) == 403, "Manifesto histórico incompleto.")
        for old in old_manifest:
            new = descriptors[old["imagem"]]
            require(new["sha256"] == old["imagem_sha256"]
                    and new["label_sha256"] == old["label_sha256"],
                    "Imagem ou anotação do teste mudou: " + old["imagem"])
        models = {}
        for model in MODELS:
            cache = source.cache(model)
            profile = source.read_json(f"test_historico/{model}/metricas_12_classes.json")
            rows = {r["id"]: r for r in profile["image_rows"]}
            require(set(rows) == set(cache) and len(cache) == 403, "TEST incompleto.")
            totals = np.zeros(3, dtype=np.int64)
            for image_id, predictions in cache.items():
                selected = [p for p in predictions if p["confianca"] >= profile["threshold"]]
                counts = point_counts(gt[image_id.rsplit("/", 1)[1]], selected)
                require(counts == tuple(rows[image_id][k] for k in ["tp", "fn", "fp"]),
                        "Ponto operacional não reproduzido: " + image_id)
                totals += counts
            require(tuple(totals) == tuple(profile["detector"][k] for k in ["tp", "fn", "fp"]),
                    "Totais operacionais divergentes: " + model)
            value, by_class = ap50(gt, cache, args.corte_previo)
            if args.corte_previo == 0.01:
                require(math.isclose(value, EXPECTED[model], abs_tol=1e-12, rel_tol=0.0),
                        "mAP50 difere da conferência anterior: " + model)
            models[model] = {
                "mAP50": value, "AP50_por_classe": by_class,
                "previsoes_capturadas": sum(map(len, cache.values())),
                "ponto_operacional": profile["detector"],
                "imagens_com_TP_FN_FP_reproduzidos": len(cache),
            }
        return {
            "fonte": {
                "arquivo": source.path.name, "sha256": sha_file(source.path),
                "objetos_test_sha256": sha_file(args.objetos_test),
                "manifesto_test_sha256": sha_file(args.manifesto_test),
            },
            "metodo": {
                "IoU": 0.5, "niveis_recall": 101, "media": "macro das 12 classes",
                "captura_conf_cache": 0.01, "corte_previo_AP": args.corte_previo,
                "limite_captura_por_imagem": 300,
                "pareamento_AP": "confiança decrescente; melhor IoU ainda disponível",
                "ponto_operacional_verificado": "pareamento histórico por IoU",
            },
            "imagens": 403, "objetos": 3465, "modelos": models,
            "referencias_mAP50_conferidas": args.corte_previo == 0.01,
            "novos_treinamentos": 0, "novas_inferencias": 0,
        }
    finally:
        source.zip.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacote-colab", required=True, type=Path)
    parser.add_argument("--objetos-test", type=Path, default=HISTORICAL / "avaliacao_objetos.csv")
    parser.add_argument("--manifesto-test", type=Path, default=HISTORICAL / "manifesto_test.json")
    parser.add_argument("--corte-previo", type=float, default=0.01,
                        help="0.01 reproduz a tabela; 0.50 é apenas diagnóstico de truncamento.")
    parser.add_argument("--saida", required=True, type=Path)
    args = parser.parse_args()
    if not 0.01 <= args.corte_previo <= 1.0:
        parser.error("--corte-previo precisa estar entre 0.01 e 1.")
    try:
        result = reconstruct(args)
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name, values in result["modelos"].items():
            print(f"{name}: mAP50={values['mAP50']:.12f} ({100*values['mAP50']:.2f}%)")
        print("TP/FN/FP reproduzidos nas 403 imagens de cada modelo; resultado:", args.saida)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print("Falha na reconstrução: " + str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
