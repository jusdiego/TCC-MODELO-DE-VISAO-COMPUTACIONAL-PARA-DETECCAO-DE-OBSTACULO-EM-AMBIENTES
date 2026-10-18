"""Entradas, auditoria documental e nova validação, sem consulta a previsões."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from diag11_core import (ALL_KNOWN, ALERT_MASK, NAMES, canonical, parse_yolo,
                         safe_member, save_csv, save_json, sha256_file)

SOURCE_URLS = {
    "classes": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "hierarquia": "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json",
    "caixas": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "rotulos": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv",
    "metadata": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}


def download(url, path, retries=3):
    """Download atômico, com recibo local e validação de tamanho quando informado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + ".download.json")
    if path.is_file() and receipt.is_file():
        prior = json.loads(receipt.read_text())
        if prior.get("url") == url and prior.get("sha256") == sha256_file(path):
            return path
    last = None
    for attempt in range(retries):
        tmp = path.with_name(path.name + ".tmp")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TCC-Diagnostico-11/1.0"})
            with urllib.request.urlopen(req, timeout=45) as response, tmp.open("wb") as out:
                expected = response.headers.get("Content-Length")
                for block in iter(lambda: response.read(2**20), b""):
                    out.write(block)
            if expected and tmp.stat().st_size != int(expected):
                raise IOError("Tamanho recebido difere do Content-Length.")
            if not tmp.stat().st_size:
                raise IOError("Arquivo recebido vazio.")
            tmp.replace(path)
            save_json(receipt, {"url": url, "sha256": sha256_file(path),
                                "bytes": path.stat().st_size})
            return path
        except Exception as exc:
            last = type(exc).__name__ + ": " + str(exc)[:180]
            tmp.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 4))
    raise IOError(f"Download não concluído: {url}; {last}")


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def load_history(csv_path, json_path, expected_hash):
    split = json.loads(Path(json_path).read_text())
    stated = split.pop("divisao_sha256")
    computed = hashlib.sha256(canonical(split).encode()).hexdigest()
    if stated != expected_hash or computed != stated:
        raise RuntimeError("Divisão histórica não corresponde à execução 10.")
    rows = list(csv_rows(csv_path))
    if len(rows) != 100000 or len({r["id"] for r in rows}) != 100000:
        raise RuntimeError("O manifesto histórico precisa de 100.000 IDs distintos.")
    if Counter(r["split"] for r in rows) != {"train": 80000, "test": 20000}:
        raise RuntimeError("Divisão histórica diferente de 80/20.")
    for label, key in [("train", "ids_treino"), ("test", "ids_teste")]:
        if {r["id"] for r in rows if r["split"] == label} != set(split[key]):
            raise RuntimeError("CSV e JSON históricos divergem.")
    hashes = {r["imagem_sha256"] for r in rows}
    if len(hashes) != len(rows):
        raise RuntimeError("O histórico contém imagens repetidas em bytes.")
    return {"ids": {r["id"] for r in rows}, "hashes": hashes, "rows": rows,
            "split_hash": stated}


def phash(path):
    with Image.open(path) as image:
        small = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
                           dtype=float)
    low = dctn(small, norm="ortho")[:8, :8]
    bits = low > np.median(low.flatten()[1:])
    bits[0, 0] = False
    return int.from_bytes(np.packbits(bits.flatten()).tobytes(), "big")


class NearIndex:
    """pHash de 64 bits, distância <=4; sinal de semelhança, não prova semântica."""
    def __init__(self):
        self.tables = [defaultdict(list) for _ in range(8)]

    def add(self, value, name):
        for band in range(8):
            self.tables[band][(value >> (8*band)) & 255].append((value, name))

    def find(self, value, max_distance=4):
        checked = set()
        for band in range(8):
            for prior, name in self.tables[band].get((value >> (8*band)) & 255, []):
                if (prior, name) in checked:
                    continue
                checked.add((prior, name))
                if (prior ^ value).bit_count() <= max_distance:
                    return name
        return None


def extract_home(zip_path, destination, expected_zip, expected_test, expected_policy):
    if sha256_file(zip_path) != expected_zip:
        raise RuntimeError("ZIP Home diferente da entrada preservada.")
    destination = Path(destination)
    records, hashes, near = [], set(), NearIndex()
    with zipfile.ZipFile(zip_path) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise RuntimeError("Há membros repetidos no ZIP Home.")
        manifest = json.loads(z.read("verificacao_preparacao.json"))
        if (manifest["test"]["sha256"] != expected_test or
                manifest["test"]["imagens"] != 403 or
                {int(k): v for k, v in manifest["classes"].items()} != dict(enumerate(NAMES))):
            raise RuntimeError("Manifesto Home incompatível.")
        root = manifest["dataset_dir"]
        items = {r["arquivo"]: r for r in manifest["arquivos"]}
        if len(items) != len(manifest["arquivos"]):
            raise RuntimeError("Manifesto Home com membros repetidos.")
        for name in z.namelist():
            safe_member(name)
            if name.startswith("codigo/") and name.endswith(".py"):
                raw = z.read(name)
                if name in items and hashlib.sha256(raw).hexdigest() != items[name]["sha256"]:
                    raise RuntimeError("Código Home diferente do manifesto.")
                out = destination / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(raw)
        for name, item in sorted(items.items()):
            is_image = (name.startswith(f"{root}/images/") and
                        Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
            if is_image:
                hashes.add(item["sha256"])
            role = None
            if name.startswith(f"{root}/images/train/raw_train/"):
                role = "home_train"
            elif name.startswith(f"{root}/images/val/raw_train/"):
                role = "home_val"
            if not is_image or role is None:
                continue
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Imagem Home alterada: {name}")
            image = destination / name
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(raw)
            label_name = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if label_name not in items:
                raise RuntimeError("Label Home ausente do manifesto.")
            label_raw = z.read(label_name)
            if hashlib.sha256(label_raw).hexdigest() != items[label_name]["sha256"]:
                raise RuntimeError("Label Home alterado.")
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            gt = parse_yolo(label_raw, width, height, source=label_name)
            record = {"id": "home:" + name, "name": name.split("/raw_train/")[-1],
                      "image": image, "role": role, "coorte": "home_val_historica",
                      "width": width, "height": height, "known_mask": ALL_KNOWN,
                      "sha256": item["sha256"], "label_sha256": items[label_name]["sha256"],
                      "gt": gt}
            record["phash"] = phash(image)
            near.add(record["phash"], name)
            records.append(record)
    counts = Counter(r["role"] for r in records)
    if counts != {"home_train": 1824, "home_val": 456}:
        raise RuntimeError(f"Home inesperado: {dict(counts)}")
    train_hash = {r["sha256"] for r in records if r["role"] == "home_train"}
    val_hash = {r["sha256"] for r in records if r["role"] == "home_val"}
    if train_hash & val_hash:
        raise RuntimeError("Home TRAIN e VAL atuais compartilham imagens em bytes.")
    policy_file = destination / "codigo/src/politica_alertas.py"
    if not policy_file.is_file() or sha256_file(policy_file) != expected_policy:
        raise RuntimeError("Política V2 diferente da versão preservada.")
    audit = {"home_train_atual": 1824, "home_val_atual": 456,
             "test_home_preservado": 403, "test_extraido": False,
             "intersecao_bytes_home_train_val_atual": 0,
             "divergencia_documental_1822_1824": "Não reconciliada; o pacote atual tem 1824.",
             "separacao_atual_nao_comprova_historico_baseline": True}
    return records, hashes, near, audit


def audit_baseline_metadata(checkpoint, expected_hash, home_records, evidence=None):
    if sha256_file(checkpoint) != expected_hash:
        raise RuntimeError("Peso baseline diferente da referência.")
    # Só o checkpoint autenticado pelo SHA esperado é desserializado.
    import torch
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = obj.get("train_args", {})
    fields = ["data", "epochs", "batch", "imgsz", "seed", "optimizer", "lr0",
              "project", "name", "fraction"]
    audit = {"modelo_sha256": expected_hash, "data_checkpoint": str(obj.get("date")),
             "train_args_preservados": {k: args[k] for k in fields if k in args},
             "estado_proveniencia_original": "nao_determinado",
             "home_val_usada_no_treino_original": None,
             "motivo": "Caminhos e argumentos não contêm a lista de imagens vistas no treino original."}
    del obj
    if evidence is not None:
        required = {"modelo_sha256", "train_image_sha256", "val_image_sha256"}
        if not required <= evidence.keys() or evidence["modelo_sha256"] != expected_hash:
            audit["evidencia_original"] = "incompativel_ou_sem_vinculo_ao_peso"
        else:
            train = set(evidence["train_image_sha256"])
            val = set(evidence["val_image_sha256"])
            valid_sha = lambda x: isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)
            if not train or any(not valid_sha(h) for h in train | val):
                raise ValueError("Manifesto original precisa de hashes SHA-256 válidos.")
            current = {r["sha256"] for r in home_records if r["role"] == "home_val"}
            overlap = train & current
            complete = evidence.get("manifesto_treino_completo") is True
            audit.update(estado_proveniencia_original="manifesto_documental_analisado",
                         evidencia_original="documento_vinculado_ao_sha_do_modelo",
                         home_val_usada_no_treino_original=True if overlap else (False if complete else None),
                         manifesto_declarado_completo=complete,
                         imagens_home_val_no_manifesto_train_original=len(overlap),
                         imagens_home_val_no_manifesto_val_original=len(current & val),
                         motivo="Conclusão limitada à completude e correção do manifesto apresentado.")
    return audit


def stable_order(iid, seed=42):
    return hashlib.sha256(f"{seed}|{iid}".encode()).hexdigest(), iid


def balanced_ids(records, n, seed=42, per_class=45):
    """Diversidade por classe e preenchimento determinístico, sem usar previsões."""
    if len(records) < n:
        raise ValueError(f"Há somente {len(records)} imagens elegíveis; exigidas {n}.")
    groups = {c: sorted((iid for iid, r in records.items() if
                         r["positive_mask"] & (1 << c)), key=lambda x: stable_order(x, seed))
              for c in range(12)}
    order = sorted(groups, key=lambda c: (len(groups[c]), c))
    chosen, counts = set(), Counter()
    # Diversificar exemplos negativos quando as seis classes têm ausência verificada.
    negatives = sorted((iid for iid, r in records.items() if
                        r["known_mask"] & ALERT_MASK == ALERT_MASK and
                        not r["positive_mask"] & ALERT_MASK), key=lambda x: stable_order(x, seed))
    for iid in negatives[:min(len(negatives), n//10)]:
        chosen.add(iid)
        counts.update(c for c in range(12) if records[iid]["positive_mask"] & (1 << c))
    for target in range(1, per_class+1):
        for cid in order:
            if len(chosen) >= n:
                break
            if counts[cid] >= target:
                continue
            for iid in groups[cid]:
                if iid not in chosen:
                    chosen.add(iid)
                    counts.update(c for c in range(12) if records[iid]["positive_mask"] & (1 << c))
                    break
    for iid in sorted(records, key=lambda x: stable_order(x, seed)):
        if len(chosen) == n:
            break
        chosen.add(iid)
    return sorted(chosen, key=lambda x: stable_order(x, seed))


def split_pool(records, n_reserve, seed=42):
    """Estratos pela classe positiva mais rara; a reserva não entra na inferência."""
    frequencies = Counter(c for r in records.values() for c in range(12)
                          if r["positive_mask"] & (1 << c))
    groups = defaultdict(list)
    for iid, record in records.items():
        classes = record["classes_positivas_avaliaveis"]
        primary = min(classes, key=lambda c: (frequencies[c], c)) if classes else -1
        groups[primary].append(iid)
    total = len(records)
    if not 0 < n_reserve < total:
        raise ValueError("A reserva precisa ser menor que o conjunto novo.")
    quotas = {c: len(ids)*n_reserve//total for c, ids in groups.items()}
    order = sorted(groups, key=lambda c: (-(len(groups[c])*n_reserve % total), c))
    for cid in order[:n_reserve-sum(quotas.values())]:
        quotas[cid] += 1
    selected = set()
    for cid, ids in groups.items():
        selected.update(sorted(ids, key=lambda x: stable_order(x, seed+1))[:quotas[cid]])
    if len(selected) != n_reserve:
        raise RuntimeError("A divisão não produziu a reserva planejada.")
    return {iid: ("reserva_futura" if iid in selected else "val_nova") for iid in records}


def filter_metadata(records, metadata_path, prior_ids):
    metadata, exclusions = {}, Counter()
    for row in csv_rows(metadata_path):
        iid = row["ImageID"]
        if iid not in records:
            continue
        if row["Subset"] != "validation":
            raise ValueError("A metadata contém uma origem diferente de validation.")
        if iid in prior_ids:
            exclusions["id_presente_nos_100k"] += 1
            continue
        try:
            rotation = float(row["Rotation"])
        except (TypeError, ValueError):
            rotation = float("nan")
        # Evita suposições sobre transformação das caixas e orientação do JPEG.
        if rotation != 0:
            exclusions["rotacao_nao_zero_ou_desconhecida"] += 1
            continue
        metadata[iid] = {k: row.get(k, "") for k in
                         ["ImageID", "OriginalLandingURL", "License", "Author", "Rotation"]}
    result = {iid: {**r, "creditos": metadata[iid]} for iid, r in records.items()
              if iid in metadata}
    return result, dict(exclusions)


def download_images(candidates, ids, directory, workers=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def one(iid):
        record = dict(candidates[iid])
        path = directory / f"oi_val_{iid}.jpg"
        try:
            download(f"https://open-images-dataset.s3.amazonaws.com/validation/{iid}.jpg", path)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                orientation = image.getexif().get(274, 1)
                if width < 32 or height < 32 or orientation != 1:
                    raise ValueError("JPEG pequeno ou com orientação EXIF não trivial.")
                image.load()
            tiny = {g["classe_id"] for g in record["gt_normalizado"] if
                    (g["bbox_norm"][2]-g["bbox_norm"][0])*width < 1 or
                    (g["bbox_norm"][3]-g["bbox_norm"][1])*height < 1}
            # A avaliação da ausência fica desconhecida para a classe afetada.
            record["restricoes_cobertura"] = dict(record["restricoes_cobertura"])
            for cid in tiny:
                for key in ["known_mask", "positive_mask", "negative_mask"]:
                    record[key] &= ~(1 << cid)
                record["restricoes_cobertura"][str(cid)] = ["caixa_menor_que_um_pixel"]
            record["classes_positivas_avaliaveis"] = [c for c in range(12)
                                                      if record["positive_mask"] & (1 << c)]
            if not (record["positive_mask"] or record["known_mask"] & ALERT_MASK == ALERT_MASK):
                raise ValueError("Nenhuma cobertura útil após conferir as dimensões reais.")
            record.update(image=path, width=width, height=height, sha256=sha256_file(path),
                          phash=phash(path), name=path.name)
            return record, None
        except Exception as exc:
            return None, {"id": iid, "motivo": type(exc).__name__, "detalhe": str(exc)[:240]}

    valid, failures = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for done, (record, failure) in enumerate(executor.map(one, ids), 1):
            if record:
                valid[record["id"]] = record
            else:
                failures.append(failure)
            if done % 100 == 0 or done == len(ids):
                print(f"Novas imagens: {done}/{len(ids)}; válidas: {len(valid)}", flush=True)
    return valid, failures


def deduplicate(records, prior_hashes, home_near, seed=42):
    accepted, excluded, seen_hash, near = {}, [], set(), NearIndex()
    for iid in sorted(records, key=lambda x: stable_order(x, seed)):
        record = records[iid]
        reason, similar = None, None
        if record["sha256"] in prior_hashes:
            reason = "bytes_presentes_nos_dados_anteriores"
        elif record["sha256"] in seen_hash:
            reason = "bytes_repetidos_no_conjunto_novo"
        elif (similar := home_near.find(record["phash"])) is not None:
            reason = "semelhanca_phash_com_home_train_ou_val"
        elif (similar := near.find(record["phash"])) is not None:
            reason = "semelhanca_phash_dentro_do_conjunto_novo"
        if reason:
            excluded.append({"id": iid, "motivo": reason, "semelhante_a": similar})
            continue
        accepted[iid] = record
        seen_hash.add(record["sha256"])
        near.add(record["phash"], iid)
    return accepted, excluded


def freeze_dataset(records, roles, directory, source_hashes, prior_split_hash, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest, runtime = [], []
    for iid in sorted(roles):
        r, role = records[iid], roles[iid]
        boxes = []
        for g in r["gt_normalizado"]:
            x1, y1, x2, y2 = g["bbox_norm"]
            boxes.append({**g, "bbox": [x1*r["width"], y1*r["height"],
                                       x2*r["width"], y2*r["height"]]})
        coverage = {k: r[k] for k in ["known_mask", "positive_mask", "negative_mask",
                                      "restricoes_cobertura"]}
        label = directory / "labels" / role / f"oi_val_{iid}.txt"
        coverage_file = directory / "coverage" / role / f"oi_val_{iid}.json"
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for g in r["gt_normalizado"]:
            x1, y1, x2, y2 = g["bbox_norm"]
            lines.append(f"{g['classe_id']} {(x1+x2)/2:.10f} {(y1+y2)/2:.10f} {(x2-x1):.10f} {(y2-y1):.10f}")
        label.write_text("\n".join(lines)+("\n" if lines else ""), encoding="utf-8")
        save_json(coverage_file, coverage)
        image_rel = f"images/{role}/oi_val_{iid}.jpg"
        item = {"id": iid, "role": role, "image": image_rel,
                "label": label.relative_to(directory).as_posix(),
                "coverage": coverage_file.relative_to(directory).as_posix(),
                "sha256": r["sha256"], "label_sha256": sha256_file(label),
                "coverage_sha256": sha256_file(coverage_file),
                "width": r["width"], "height": r["height"], "known_mask": r["known_mask"],
                "positive_mask": r["positive_mask"], "negative_mask": r["negative_mask"],
                "phash": f"{r['phash']:016x}", "creditos": r["creditos"],
                "gt_normalizado": r["gt_normalizado"]}
        manifest.append(item)
        runtime.append({**r, "role": role, "coorte": "openimages_val_nova",
                        "gt": boxes, "label": label, "coverage": coverage_file})
    metadata = {"versao": "11-validacao-independente-v1", "seed": seed, "nomes": NAMES,
                "fontes_metadata_sha256": source_hashes, "origem_split": "validation",
                "divisao_100k_anterior_sha256": prior_split_hash,
                "contagens": dict(Counter(roles.values())),
                "criterio": "Diversidade de classes, hash determinístico e pHash; sem previsões",
                "cobertura": "Positivo explícito na raiz ou negativo na raiz/ancestral; classes problemáticas excluídas",
                "independencia_limitada_ao_historico_conhecido": True,
                "pretreinamentos_publicos_nao_auditados": True,
                "reserva_futura_avaliada": False, "arquivos": manifest}
    dataset_hash = hashlib.sha256(canonical(metadata).encode()).hexdigest()
    metadata["dataset_sha256"] = dataset_hash
    save_json(directory / "manifesto_validacao_nova.json", metadata)
    save_csv(directory / "divisao.csv", [{k: r[k] for k in
                                        ["id", "role", "sha256", "known_mask", "positive_mask"]}
                                       for r in manifest])
    archive = directory.parent / f"validacao_independente_11_{dataset_hash[:12]}.zip"
    # Os JPEGs permanecem sem recompressão.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.write(directory / "manifesto_validacao_nova.json", "manifesto_validacao_nova.json")
        z.write(directory / "divisao.csv", "divisao.csv")
        for r, item in zip(sorted(runtime, key=lambda r: r["id"]), manifest):
            z.write(r["image"], item["image"])
            z.write(r["label"], item["label"])
            z.write(r["coverage"], item["coverage"])
    return runtime, metadata, archive


def support_rows(records):
    rows = []
    for role in sorted({r["role"] for r in records}):
        group = [r for r in records if r["role"] == role]
        for cid, name in enumerate(NAMES):
            gt = [g for r in group for g in r["gt"] if
                  g["classe_id"] == cid and r["known_mask"] & (1 << cid)]
            rows.append({"conjunto": role, "classe_id": cid, "classe": name,
                         "imagens_cobertura_conhecida": sum(bool(r["known_mask"] & (1 << cid)) for r in group),
                         "imagens_positivas_avaliaveis": sum(bool(r["positive_mask"] & (1 << cid)) for r in group),
                         "caixas_avaliaveis": len(gt), "suporte_reduzido": len(gt) < 30})
    return rows
