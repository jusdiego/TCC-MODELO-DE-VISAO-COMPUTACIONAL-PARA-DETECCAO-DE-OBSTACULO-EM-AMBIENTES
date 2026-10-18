"""Entradas congeladas e replay somente do TRAIN histórico."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import yaml

import diag11_core as core
import diag11_data as d11
from exp12_protocol import diverse_subset


def check_digest(raw, expected, label):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("Conteúdo alterado: " + label)


def materialise_home_labels(zip_path, home):
    with zipfile.ZipFile(zip_path) as z:
        for record in home:
            member = record["id"].removeprefix("home:")
            label_name = member.replace("/images/", "/labels/").rsplit(".", 1)[0]+".txt"
            core.safe_member(label_name)
            raw = z.read(label_name)
            check_digest(raw, record["label_sha256"], label_name)
            label = Path(str(record["image"]).replace("/images/", "/labels/")).with_suffix(".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_bytes(raw)
            record["label"] = label


def read_package11(path, expected_sha, expected_models, expected_dataset):
    if core.sha256_file(path) != expected_sha:
        raise RuntimeError("O pacote do diagnóstico 11 mudou.")
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise RuntimeError("Pacote 11 com membros repetidos.")
        for name in z.namelist():
            core.safe_member(name)
        summary = json.loads(z.read("resumo_diagnostico_11.json"))
        config = json.loads(z.read("configuracao_diagnostico.json"))
        dataset_location = json.loads(z.read("localizacao_dataset.json"))
        manifest = json.loads(z.read("manifesto_validacao_nova.json"))
        if (config["modelos_sha256"] != expected_models or
                config["dataset_sha256"] != expected_dataset or
                dataset_location["dataset_sha256"] != expected_dataset or
                manifest["dataset_sha256"] != expected_dataset or
                summary["reserva_futura_inferida"] is not False):
            raise RuntimeError("Identidades ou reserva do diagnóstico 11 incompatíveis.")
    return summary, config, dataset_location, manifest


def load_frozen_validation(zip_path, destination, expected_archive_sha, expected_dataset_sha,
                           expected_manifest):
    if core.sha256_file(zip_path) != expected_archive_sha:
        raise RuntimeError("ZIP da validação independente diferente do congelado no 11.")
    destination = Path(destination)
    val, reserved_ids, frozen_files = [], set(), []
    with zipfile.ZipFile(zip_path) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise RuntimeError("Dataset novo com membros repetidos.")
        for member in z.namelist():
            core.safe_member(member)
        manifest = json.loads(z.read("manifesto_validacao_nova.json"))
        if manifest != expected_manifest:
            raise RuntimeError("O manifesto do pacote difere do diagnóstico 11.")
        raw_manifest = dict(manifest)
        claimed = raw_manifest.pop("dataset_sha256")
        if hashlib.sha256(core.canonical(raw_manifest).encode()).hexdigest() != claimed or claimed != expected_dataset_sha:
            raise RuntimeError("Hash canônico da validação nova não confere.")
        if manifest["nomes"] != core.NAMES or manifest["origem_split"] != "validation":
            raise RuntimeError("Classes ou origem da validação alteradas.")
        items = manifest["arquivos"]
        if (Counter(r["role"] for r in items) != {"val_nova": 1000, "reserva_futura": 300}
                or len({r["id"] for r in items}) != 1300
                or len({r["sha256"] for r in items}) != 1300):
            raise RuntimeError("Identidades ou quantidades do conjunto independente incompatíveis.")
        for item in items:
            role = item["role"]
            for field, prefix in [("image", "images"), ("label", "labels"), ("coverage", "coverage")]:
                if not item[field].startswith(prefix+"/"+role+"/"):
                    raise RuntimeError("Caminho do dataset não corresponde ao papel declarado.")
                if item[field] not in z.namelist():
                    raise RuntimeError("Arquivo declarado ausente no ZIP.")
            if role == "reserva_futura":
                reserved_ids.add(item["id"])
                continue  # Nenhum JPEG, label ou cobertura da reserva é extraído.
            paths = {}
            for field, hash_field in [("image", "sha256"), ("label", "label_sha256"),
                                      ("coverage", "coverage_sha256")]:
                raw = z.read(item[field])
                check_digest(raw, item[hash_field], item[field])
                output = destination/item[field]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(raw)
                paths[field] = output
                frozen_files.append((output, item[hash_field]))
            coverage = json.loads(paths["coverage"].read_text())
            if any(coverage[k] != item[k] for k in ["known_mask", "positive_mask", "negative_mask"]):
                raise RuntimeError("Cobertura da validação foi alterada.")
            with Image.open(paths["image"]) as image:
                if image.size != (item["width"], item["height"]):
                    raise RuntimeError("Dimensões diferentes do manifesto.")
                image.verify()
            parsed = core.parse_yolo(paths["label"].read_bytes(), item["width"], item["height"])
            if len(parsed) != len(item["gt_normalizado"]):
                raise RuntimeError("Quantidade de caixas difere do manifesto.")
            gt = []
            for original, label_obj in zip(item["gt_normalizado"], parsed):
                a, b, c, d = original["bbox_norm"]
                box = [a*item["width"], b*item["height"], c*item["width"], d*item["height"]]
                if (original["classe_id"] != label_obj["classe_id"] or
                        not np.allclose(box, label_obj["bbox"], rtol=0, atol=.001)):
                    raise RuntimeError("Caixa do manifesto diverge do label autenticado.")
                gt.append({**original, "bbox": box})
            val.append({**item, **paths, "gt": gt, "role": "val_nova",
                        "coorte": "openimages_val_nova", "name": paths["image"].name})
    return val, reserved_ids, frozen_files


def load_replay(package_items, selected_history, local_dir, fetch_package, expected_cfg,
                forbidden_ids, forbidden_hashes, protected_near, validator, final_size=2000, seed=42):
    """Lê só os membros escolhidos dos cinco lotes existentes, com SHA integral."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    wanted = {r["id"]: r for r in selected_history}
    protected_ids, protected_hashes = set(forbidden_ids), set(forbidden_hashes)
    records, exclusions = {}, []
    frozen, seen, package_receipts = [], set(), []
    for item in package_items:
        pkg = local_dir/item["arquivo"]
        fetch_package(item, pkg)
        if pkg.stat().st_size != item["zip_bytes"] or core.sha256_file(pkg) != item["zip_sha256"]:
            raise RuntimeError("Pacote de replay incompleto ou alterado.")
        with zipfile.ZipFile(pkg) as z:
            if len(z.namelist()) != len(set(z.namelist())):
                raise RuntimeError("Lote de replay com membros repetidos.")
            for name in z.namelist():
                core.safe_member(name)
            manifest = json.loads(z.read("manifesto_lote.json"))
            if manifest["config_sha256"] != expected_cfg or manifest["origem_split"] != "train":
                raise RuntimeError("Lote de replay não pertence ao TRAIN autenticado.")
            for r in manifest["aceitas"]:
                if r["id"] not in wanted:
                    continue
                if r["id"] in seen:
                    raise RuntimeError("ID de replay repetido entre lotes.")
                seen.add(r["id"])
                prior = wanted[r["id"]]
                if (prior["split"] != "train" or prior["lote"] != item["arquivo"] or
                        prior["imagem_sha256"] != r["sha256"] or
                        int(prior["known_mask"]) != r["known_mask"]):
                    raise RuntimeError("Replay diverge da divisão 80/20 histórica.")
                if r["id"] in protected_ids or r["sha256"] in protected_hashes:
                    raise RuntimeError("Replay coincide com dado protegido.")
                raw_image, raw_label, raw_cov = (z.read(r[k]) for k in ["imagem", "label", "cobertura"])
                for raw, key in [(raw_image, "sha256"), (raw_label, "label_sha256"), (raw_cov, "cobertura_sha256")]:
                    check_digest(raw, r[key], r["id"]+"/"+key)
                classes = validator(raw_label, source=item["arquivo"]+":"+r["label"])
                cov = json.loads(raw_cov)
                known_classes = (set(cov["classes_positivas_verificadas_exaustivas"]) |
                                 set(cov["classes_negativas_verificadas"]))
                if (not all(isinstance(cid, int) and 0 <= cid < 12 for cid in known_classes)
                        or sum(1 << cid for cid in known_classes) != r["known_mask"]
                        or cov["known_mask"] != r["known_mask"]
                        or set(classes) & set(cov["classes_negativas_verificadas"])):
                    raise RuntimeError("Máscara ou anotação parcial de replay incompatível.")
                image = local_dir/"images/train"/f"oi_{r['id']}.jpg"
                label = local_dir/"labels/train"/f"oi_{r['id']}.txt"
                coverage_file = local_dir/"coverage/train"/f"oi_{r['id']}.json"
                for path, raw in [(image, raw_image), (label, raw_label), (coverage_file, raw_cov)]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                with Image.open(image) as im:
                    width, height = im.size
                    im.verify()
                if (width, height) != (r["width"], r["height"]):
                    raise RuntimeError("Dimensões de replay incompatíveis.")
                near = protected_near.find(d11.phash(image))
                if near:
                    exclusions.append({"id": r["id"], "motivo": "semelhanca_phash_com_validacao_ou_reserva", "semelhante_a": near})
                    continue
                gt = core.parse_yolo(raw_label, width, height)
                records[r["id"]] = {
                    **prior, "oi_id": r["id"], "id": r["id"], "source": "openimages",
                    "image": image, "label": label, "coverage": coverage_file,
                    "known_mask": r["known_mask"], "sha256": r["sha256"],
                    "label_sha256": r["label_sha256"], "coverage_sha256": r["cobertura_sha256"],
                    "width": width, "height": height, "classes_com_caixa": classes,
                    "gt": gt, "name": image.name, "role": "train", "coorte": "replay_train",
                }
        package_receipts.append(item)
        pkg.unlink()
    if seen != set(wanted):
        raise RuntimeError("Alguns IDs planejados não foram encontrados nos cinco lotes.")
    if len(records) < final_size:
        raise RuntimeError(f"Replay com somente {len(records)} elegíveis; exigidos {final_size}.")
    chosen = diverse_subset(list(records.values()), final_size, seed)
    output = {Path(r["image"]).resolve().as_posix(): r for r in chosen}
    for record in output.values():
        frozen += [(record["image"], record["sha256"]), (record["label"], record["label_sha256"]),
                   (record["coverage"], record["coverage_sha256"])]
    return output, exclusions, frozen, package_receipts


def make_training_inputs(home, replay, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    home_train = {Path(r["image"]).resolve().as_posix():
                  {**r, "source": "home", "known_mask": core.ALL_KNOWN}
                  for r in home if r["role"] == "home_train"}
    val = [Path(r["image"]).resolve().as_posix() for r in home if r["role"] == "home_val"]
    if len(home_train) != 1824 or len(val) != 456:
        raise RuntimeError("Home diferente da configuração congelada.")
    val_file = destination/"home_val.txt"
    val_file.write_text("\n".join(sorted(val))+"\n", encoding="utf-8")
    arms = {}
    for name, records in [("A_home", home_train), ("B_home_replay", {**home_train, **replay})]:
        path = destination/(name+".txt")
        path.write_text("\n".join(sorted(records))+"\n", encoding="utf-8")
        data_file = destination/(name+".yaml")
        data_file.write_text(yaml.safe_dump({"path": str(destination.resolve()), "train": str(path.resolve()),
                                           "val": str(val_file.resolve()), "names": dict(enumerate(core.NAMES))},
                                          sort_keys=False, allow_unicode=True), encoding="utf-8")
        arms[name] = {"records": records, "yaml": data_file}
    return arms


def restore_result(package_path, cohort, model, cache_dir):
    cache_dir = Path(cache_dir)/cohort/model
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path) as z:
        prefix = cohort+"/"+model+"/"
        summary = json.loads(z.read(prefix+"metricas.json"))
        def rows(name):
            return list(csv.DictReader(io.StringIO(z.read(prefix+name).decode("utf-8-sig"))))
        images, events, classes = rows("alertas_por_imagem.csv"), rows("avaliacao_objetos.csv"), rows("metricas_por_classe.csv")
        for row in images:
            for k in ["tp_alerta", "fn_alerta", "fp_alerta", "alta_ref", "alta_sem_alerta", "tp_detector",
                      "fn_detector", "fp_detector", "gt_avaliaveis", "gt_ignorados", "predicoes_ignoradas_conf_025"]:
                row[k] = int(row[k])
            for k in ["cobertura_completa_alertas", "ha_alerta_ref", "ha_alerta_pred"]:
                row[k] = row[k] == "True"
        for row in events:
            row["classe_id"] = int(row["classe_id"])
            for k in ["ha_alerta_ref", "ha_alerta_pred"]:
                row[k] = row[k] == "True"
        cache = cache_dir/"predicoes_conf_0001.jsonl.gz"
        cache.write_bytes(z.read(prefix+"predicoes_conf_0001.jsonl.gz"))
    return {"summary": summary, "images": images, "events": events, "classes": classes, "cache_file": cache}


def verify_frozen(files):
    for path, sha in files:
        if core.sha256_file(path) != sha:
            raise RuntimeError("Arquivo protegido alterado: "+str(path))
