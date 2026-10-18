"""Experimento 09: caixas humanas parciais, fontes equilibradas e Home VAL."""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ultralytics
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Sampler
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model


VERSION = "10-openimages-80-20-1ep-3.0"
ULTRALYTICS_VERSION = "8.4.150"
LOSS_SOURCE_SHA = "357053aa9a24ad2c4d541c3ada3a27232a10d679f89c7b4957b9a84a7da4ec1d"

NOMES = [
    "bed", "sofa", "chair", "table", "lamp", "tv",
    "laptop", "wardrobe", "window", "door",
    "potted plant", "photo frame"
]

ALL_KNOWN = (1 << 12) - 1


def sha256_file(p):
    h = hashlib.sha256()

    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)

    return h.hexdigest()


def canonical_json(x):
    return json.dumps(
        x,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    )


def save_json(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps(x, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp.replace(p)


def check_api():
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Instale ultralytics=={ULTRALYTICS_VERSION} "
            "e reinicie o ambiente se necessário."
        )

    if sha256_file(inspect.getfile(v8DetectionLoss)) != LOSS_SOURCE_SHA:
        raise RuntimeError(
            "A implementação da perda difere da versão verificada; "
            "interrompido antes do treino."
        )


class BCEComCobertura(nn.Module):
    """Zera perdas negativas de classes desconhecidas; conserva positivos."""

    def __init__(self, original=None):
        super().__init__()
        self.original = original or nn.BCEWithLogitsLoss(reduction="none")
        self.known = None

    def forward(self, logits, target):
        if self.known is None:
            raise RuntimeError(
                "Cobertura não foi transmitida à perda parcial."
            )

        known = self.known.to(
            device=logits.device,
            dtype=torch.bool
        )

        if known.shape != (logits.shape[0], logits.shape[-1]):
            raise RuntimeError(
                "Dimensões de cobertura incompatíveis com batch/classes."
            )

        permitted = known[:, None, :] | (target > 0)

        return self.original(logits, target) * permitted.to(logits.dtype)


class DetectionLossParcial(v8DetectionLoss):
    """Altera somente a BCE; caixa, DFL, assignment e ganhos são os padrões."""

    def __init__(self, model):
        super().__init__(model)

        if (
            self.nc != 12
            or not isinstance(self.bce, nn.BCEWithLogitsLoss)
            or self.bce.reduction != "none"
        ):
            raise RuntimeError(
                "Cabeça ou perda de detecção não é a esperada "
                "para este experimento."
            )

        self.bce = BCEComCobertura(self.bce)

    def loss(self, preds, batch):
        if "class_known" not in batch:
            raise RuntimeError(
                "Batch de treinamento sem cobertura; "
                "não substituir por negativos presumidos."
            )

        self.bce.known = batch["class_known"]

        try:
            return super().loss(preds, batch)
        finally:
            self.bce.known = None


class SamplerDuasFontes(Sampler):
    """Cada batch contém metade Home e metade OI; uma época cobre todos os OI."""

    def __init__(self, home_indices, oi_indices, batch_size, seed=42):
        self.home = list(home_indices)
        self.oi = list(oi_indices)

        if (
            not self.home
            or not self.oi
            or batch_size < 2
            or batch_size % 2
        ):
            raise ValueError(
                "São necessárias duas fontes e um batch par >=2."
            )

        self.batch = int(batch_size)
        self.half = self.batch // 2
        self.batches = math.ceil(len(self.oi) / self.half)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches * self.batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        home, oi = [], []
        needed = self.batches * self.half

        while len(home) < needed:
            home.extend(rng.permutation(self.home).tolist())

        while len(oi) < needed:
            oi.extend(rng.permutation(self.oi).tolist())

        for start in range(0, needed, self.half):
            batch = (
                home[start:start + self.half]
                + oi[start:start + self.half]
            )
            rng.shuffle(batch)
            yield from batch


class LoaderFinito(DataLoader):
    def reset(self):
        # Um novo iterador é criado em cada época.
        pass


def epoch_sampler(trainer):
    sampler = getattr(trainer.train_loader, "sampler", None)

    if isinstance(sampler, SamplerDuasFontes):
        sampler.set_epoch(trainer.epoch)


class TrainerOpenImages(DetectionTrainer):
    # Configurado antes de criar o trainer.
    # Não é anexado ao DetectionModel salvo.
    source_records = None

    def __init__(self, *args, **kwargs):
        check_api()
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", epoch_sampler)

    def get_dataloader(
        self,
        dataset_path,
        batch_size=16,
        rank=0,
        mode="train"
    ):
        if mode != "train":
            return super().get_dataloader(
                dataset_path,
                batch_size,
                rank,
                mode
            )

        if self.source_records is None:
            raise RuntimeError(
                "As fontes/coberturas ainda não foram configuradas."
            )

        if self.world_size > 1 or self.args.compile or self.args.rect:
            raise RuntimeError(
                "Este experimento usa uma GPU, "
                "compile=False e rect=False."
            )

        for k in ["mosaic", "mixup", "cutmix", "copy_paste"]:
            if float(getattr(self.args, k, 0)) != 0:
                raise RuntimeError(
                    f"{k} deve ficar em zero para preservar "
                    "a cobertura por imagem."
                )

        dataset = self.build_dataset(
            dataset_path,
            mode,
            batch_size
        )

        home, oi = [], []

        for i, arquivo in enumerate(dataset.im_files):
            key = Path(arquivo).resolve().as_posix()

            if key not in self.source_records:
                raise RuntimeError(
                    f"Imagem sem proveniência/cobertura: {arquivo}"
                )

            source = self.source_records[key]["source"]

            if source == "home":
                home.append(i)
            elif source == "openimages":
                oi.append(i)
            else:
                raise RuntimeError(
                    "Fonte de treinamento não reconhecida."
                )

        sampler = SamplerDuasFontes(
            home,
            oi,
            batch_size,
            self.args.seed
        )

        generator = torch.Generator().manual_seed(self.args.seed)

        return LoaderFinito(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=dataset.collate_fn,
            generator=generator,
            drop_last=False
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        masks = []

        for arquivo in batch["im_file"]:
            key = Path(arquivo).resolve().as_posix()

            if key not in self.source_records:
                raise RuntimeError(
                    "Batch contém imagem fora do manifesto de treino."
                )

            bitmask = self.source_records[key]["known_mask"]

            masks.append([
                bool(bitmask & (1 << cid))
                for cid in range(12)
            ])

        batch["class_known"] = torch.tensor(
            masks,
            dtype=torch.bool,
            device=self.device
        )

        model = unwrap_model(self.model)
        criterion = getattr(model, "criterion", None)

        if criterion is None:
            # Modelo padrão, já colocado no dispositivo.
            # A EMA foi criada antes desta etapa e valida Home
            # com a perda padrão de cobertura completa.
            model.criterion = DetectionLossParcial(model)

        elif not isinstance(criterion, DetectionLossParcial):
            raise RuntimeError(
                "O treino iniciou com uma perda diferente "
                "da perda parcial."
            )

        return batch


def self_test_bce():
    logits = torch.zeros(
        (2, 3, 12),
        requires_grad=True
    )

    targets = torch.zeros_like(logits)
    targets[1, 0, 7] = 1

    known = torch.ones(
        (2, 12),
        dtype=torch.bool
    )
    known[1, 7] = False
    known[1, 10] = False

    partial = BCEComCobertura()
    partial.known = known
    partial(logits, targets).sum().backward()

    assert logits.grad[1, :, 10].eq(0).all()
    assert logits.grad[1, 1:, 7].eq(0).all()
    assert logits.grad[1, 0, 7].item() < 0
    assert logits.grad[0].ne(0).all()

    full = BCEComCobertura()
    full.known = torch.ones(
        (2, 12),
        dtype=torch.bool
    )

    torch.testing.assert_close(
        full(logits.detach(), targets),
        nn.BCEWithLogitsLoss(reduction="none")(
            logits.detach(),
            targets
        )
    )

    return {
        "gradiente_negativo_desconhecido_zero": True,
        "caixa_positiva_preservada": True,
        "cobertura_completa_igual_bce_padrao": True
    }


def read_safe(z, name):
    path = PurePosixPath(name)

    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise RuntimeError(
            f"Caminho inseguro dentro de pacote: {name}"
        )

    return z.read(name)


def validate_yolo(raw, nclasses=12, source="<anotação>"):
    """
    Ignora linhas vazias e valida as linhas preenchidas.
    Em caso de erro, informa arquivo, linha e conteúdo.
    """
    result = []

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Anotação YOLO não está em UTF-8: {source}"
        ) from exc

    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue

        values = line.split()
        location = f"{source}, linha {line_number}"

        if len(values) != 5:
            raise RuntimeError(
                f"Anotação YOLO inválida em {location}: "
                "esperados 5 campos "
                "(classe x y largura altura), "
                f"encontrados {len(values)}. "
                f"Conteúdo: {line[:160]!r}"
            )

        try:
            cid, *bbox = map(float, values)
        except ValueError as exc:
            raise RuntimeError(
                f"Campo não numérico em {location}: "
                f"{line[:160]!r}"
            ) from exc

        if (
            not cid.is_integer()
            or not 0 <= cid < nclasses
            or not all(math.isfinite(x) for x in bbox)
        ):
            raise RuntimeError(
                f"Classe ou caixa YOLO inválida em {location}: "
                f"{line[:160]!r}"
            )

        x, y, w, h = bbox

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
            and x - w / 2 >= -1e-6
            and x + w / 2 <= 1 + 1e-6
            and y - h / 2 >= -1e-6
            and y + h / 2 <= 1 + 1e-6
        ):
            raise RuntimeError(
                f"Caixa YOLO fora dos limites em {location}: "
                f"{line[:160]!r}"
            )

        result.append(int(cid))

    if not result:
        raise RuntimeError(
            f"Imagem desta preparação sem caixas: {source}"
        )

    return result


def write_checked(dest, raw, expected_sha):
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError(
            "Conteúdo do pacote diferente do SHA-256 no manifesto."
        )

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)


def prepare_training(
    zip_home,
    oi_drive,
    local,
    results,
    expected,
    batches_oi=None,
    home_train_count=1824,
    home_val_count=456,
    oi_count=100000,
    package_fetcher=None
):
    """Snapshot Home TRAIN/VAL e de todos os lotes liberados para 100k."""

    local = Path(local)
    results = Path(results)
    oi_drive = Path(oi_drive)

    local.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    prep_cfg = json.loads(
        (oi_drive / "configuracao_preparacao.json").read_text()
    )

    cfg_sha = hashlib.sha256(
        canonical_json(prep_cfg).encode()
    ).hexdigest()

    if (
        cfg_sha != expected["oi_config_sha256"]
        or prep_cfg["nomes"] != NOMES
    ):
        raise RuntimeError(
            "Preparação Open Images diferente da auditada no piloto."
        )

    home_zip_local = local / "home_sun_snapshot.zip"
    shutil.copyfile(zip_home, home_zip_local)

    if sha256_file(home_zip_local) != expected["home_zip_sha256"]:
        raise RuntimeError(
            "ZIP Home+SUN diferente do preservado no 07b/08."
        )

    records, home_val = {}, []
    extract_home = local / "base_existente"

    with zipfile.ZipFile(home_zip_local) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise RuntimeError(
                "ZIP Home+SUN tem nomes repetidos."
            )

        manifest = json.loads(
            read_safe(z, "verificacao_preparacao.json")
        )

        if (
            manifest["test"]["imagens"] != 403
            or manifest["test"]["sha256"] != expected["test_sha256"]
        ):
            raise RuntimeError(
                "TEST preservado diferente do esperado."
            )

        if (
            {int(k): v for k, v in manifest["classes"].items()}
            != dict(enumerate(NOMES))
        ):
            raise RuntimeError(
                "Classes/IDs da base Home incompatíveis."
            )

        root = manifest["dataset_dir"]

        for item in manifest["arquivos"]:
            name = item["arquivo"]
            p = PurePosixPath(name)

            in_home_train = (
                name.startswith(f"{root}/images/train/raw_train/")
                or name.startswith(f"{root}/labels/train/raw_train/")
            )

            in_home_val = (
                name.startswith(f"{root}/images/val/raw_train/")
                or name.startswith(f"{root}/labels/val/raw_train/")
            )

            code_or_meta = (
                name.startswith("codigo/")
                or (
                    p.suffix in {
                        ".json", ".csv", ".yaml", ".yml", ".txt"
                    }
                    and "/images/" not in name
                    and "/labels/" not in name
                )
            )

            if not (in_home_train or in_home_val or code_or_meta):
                continue

            raw = read_safe(z, name)
            dest = extract_home / name
            write_checked(dest, raw, item["sha256"])

            if name.startswith(f"{root}/images/train/raw_train/"):
                records[dest.resolve().as_posix()] = {
                    "source": "home",
                    "known_mask": ALL_KNOWN,
                    "sha256": item["sha256"]
                }

            elif name.startswith(f"{root}/images/val/raw_train/"):
                home_val.append(
                    dest.resolve().as_posix()
                )

        listed = {
            item["arquivo"]
            for item in manifest["arquivos"]
        }

        # O hash do ZIP também autentica o código auxiliar
        # que não esteja listado individualmente no manifesto.
        for name in set(z.namelist()) - listed:
            if name.startswith("codigo/") and not name.endswith("/"):
                raw = read_safe(z, name)

                write_checked(
                    extract_home / name,
                    raw,
                    hashlib.sha256(raw).hexdigest()
                )

    home_dataset = extract_home / root
    home_images = list(records)

    if (
        len(home_images) != home_train_count
        or len(home_val) != home_val_count
    ):
        raise RuntimeError(
            "Home TRAIN/VAL inesperados: "
            f"{len(home_images)}/{len(home_val)}; "
            f"esperados {home_train_count}/{home_val_count}."
        )

    for image in [*home_images, *home_val]:
        label = Path(
            image.replace("/images/", "/labels/")
        ).with_suffix(".txt")

        validate_yolo(
            label.read_bytes(),
            source=str(label)
        )

    packages = json.loads(
        (oi_drive / "manifesto_pacotes.json").read_text()
    )

    if packages["config_sha256"] != cfg_sha:
        raise RuntimeError(
            "Manifesto dos lotes tem configuração diferente."
        )

    selected = sorted(
        packages["lotes"],
        key=lambda r: r["arquivo"]
    )
    if batches_oi is None:
        batches_oi = len(selected)

    if len(selected) != batches_oi:
        raise RuntimeError(
            "Quantidade de lotes difere da liberação para 100k."
        )

    oi_ids, oi_hashes, package_snapshot = set(), set(), []
    oi_root = local / "openimages"
    per_class = Counter()

    for item in selected:
        name = item["arquivo"]

        if PurePosixPath(name).name != name:
            raise RuntimeError(
                "Nome de pacote não é um nome simples."
            )

        snapshot = local / name

        if package_fetcher is None:
            shutil.copyfile(oi_drive / "lotes" / name, snapshot)
        else:
            print(f"Pacote {len(package_snapshot) + 1}/{len(selected)}: {name}", flush=True)
            package_fetcher(item, snapshot)

        if (
            sha256_file(snapshot) != item["zip_sha256"]
            or snapshot.stat().st_size != item["zip_bytes"]
        ):
            raise RuntimeError(
                f"Pacote alterado ou incompleto: {name}"
            )

        with zipfile.ZipFile(snapshot) as z:
            if len(z.namelist()) != len(set(z.namelist())):
                raise RuntimeError(
                    "ZIP Open Images tem membros repetidos."
                )

            m = json.loads(
                read_safe(z, "manifesto_lote.json")
            )

            if (
                m["config_sha256"] != cfg_sha
                or m["origem_split"] != "train"
            ):
                raise RuntimeError(
                    "Lote não é da seleção do treino "
                    "Open Images auditado."
                )

            for r in m["aceitas"]:
                if r["id"] in oi_ids or r["sha256"] in oi_hashes:
                    raise RuntimeError(
                        "ID ou bytes duplicados nos lotes usados no treino."
                    )

                oi_ids.add(r["id"])
                oi_hashes.add(r["sha256"])

                im_raw = read_safe(z, r["imagem"])
                label_raw = read_safe(z, r["label"])
                cov_raw = read_safe(z, r["cobertura"])

                for key, raw, hkey in [
                    ("imagem", im_raw, "sha256"),
                    ("label", label_raw, "label_sha256"),
                    ("cobertura", cov_raw, "cobertura_sha256")
                ]:
                    write_checked(
                        oi_root / r[key],
                        raw,
                        r[hkey]
                    )

                cls = validate_yolo(
                    label_raw,
                    source=f"{name}:{r['label']}"
                )

                cov = json.loads(cov_raw)

                if (
                    cov["known_mask"] != r["known_mask"]
                    or not isinstance(r["known_mask"], int)
                    or not 0 <= r["known_mask"] <= ALL_KNOWN
                ):
                    raise RuntimeError(
                        "Cobertura do lote inconsistente."
                    )

                known_list = (
                    set(cov["classes_positivas_verificadas_exaustivas"])
                    | set(cov["classes_negativas_verificadas"])
                )

                if sum(1 << cid for cid in known_list) != r["known_mask"]:
                    raise RuntimeError(
                        "Lista de classes conhecidas "
                        "não corresponde à máscara."
                    )

                if set(cls) & set(cov["classes_negativas_verificadas"]):
                    raise RuntimeError(
                        "Caixa positiva contradiz ausência verificada."
                    )

                with Image.open(io.BytesIO(im_raw)) as im:
                    im.load()

                    if im.size != (r["width"], r["height"]):
                        raise RuntimeError(
                            "Dimensões da imagem diferem do manifesto."
                        )

                image = (
                    oi_root / r["imagem"]
                ).resolve().as_posix()

                records[image] = {
                    "source": "openimages", "known_mask": r["known_mask"], "sha256": r["sha256"],
                    "oi_id": r["id"], "lote": name,
                    "classe_primaria": int(r.get("classe_primaria_selecao", min(cls))),
                    "classes_com_caixa": cls, "width": r["width"], "height": r["height"],
                    "label_sha256": r["label_sha256"], "cobertura_sha256": r["cobertura_sha256"],
                }

                per_class.update(cls)

        package_snapshot.append(item)

        # Remove apenas a cópia local do ZIP do lote.
        snapshot.unlink()

    if len(oi_ids) != oi_count:
        raise RuntimeError(
            f"Treinamento esperava {oi_count} imagens OI, "
            f"encontrou {len(oi_ids)}."
        )

    if set(home_val) & set(records):
        raise RuntimeError(
            "Imagem de validação aparece no treino."
        )

    val_hashes = {
        sha256_file(x)
        for x in home_val
    }

    if val_hashes & {
        r["sha256"]
        for r in records.values()
    }:
        raise RuntimeError(
            "Imagem de treino coincide em bytes "
            "com a validação Home."
        )

    # O 08 já realizou a triagem de contaminação.
    # Aqui repetimos a verificação por bytes contra Home VAL.
    train_list = local / "train.txt"
    val_list = local / "home_val.txt"

    train_list.write_text(
        "\n".join(sorted(records)) + "\n",
        encoding="utf-8"
    )

    val_list.write_text(
        "\n".join(sorted(home_val)) + "\n",
        encoding="utf-8"
    )

    data_yaml = local / "data_home_openimages.yaml"

    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(local.resolve()),
                "train": str(train_list),
                "val": str(val_list),
                "names": dict(enumerate(NOMES))
            },
            sort_keys=False,
            allow_unicode=True
        ),
        encoding="utf-8"
    )

    save_json(
        local / "fontes_cobertura_treino.json",
        records
    )

    summary = {
        "versao": VERSION,
        "home_train_imagens_unicas": len(home_images),
        "home_val_imagens": len(home_val),
        "openimages_imagens_unicas": len(oi_ids),
        "treino_imagens_unicas": len(records),
        "caixas_openimages": sum(per_class.values()),
        "caixas_openimages_por_classe": dict(per_class),
        "lotes_utilizados": package_snapshot,
        "home_zip_sha256": expected["home_zip_sha256"],
        "test_sha256": expected["test_sha256"],
        "test_imagens": 403,
        "test_extraido_ou_avaliado": False,
        "sun_incluida_no_novo_treino": False,
        "amostragem": (
            "50% Home / 50% OI em cada batch; "
            "Home reamostrado por índices, sem duplicar arquivos"
        ),
        "selecao_checkpoint": (
            "mAP50-95 Home VAL; triagem de alertas após treino"
        ),
        "tratamento_parcial": (
            "Zero gradiente BCE para negativos de classes desconhecidas; "
            "caixas positivas preservadas"
        ),
        "config_preparacao_sha256": cfg_sha
    }

    save_json(
        results / "resumo_dados_treino.json",
        summary
    )

    save_json(
        results / "lotes_utilizados.json",
        package_snapshot
    )

    TrainerOpenImages.source_records = records

    return {
        "yaml": data_yaml,
        "home_dataset": home_dataset,
        "codigo": extract_home / "codigo",
        "records": records,
        "home_val": home_val,
        "resumo": summary
    }


def export_standard_checkpoint(source, dest):
    """Confere que o checkpoint não depende das classes do treinador parcial."""

    obj = torch.load(
        source,
        map_location="cpu",
        weights_only=False
    )

    model = obj.get("ema") or obj.get("model")

    if type(model) is not DetectionModel:
        raise RuntimeError(
            "O checkpoint não contém DetectionModel padrão."
        )

    model.criterion = None
    model = model.float().eval()

    if not all(
        torch.isfinite(t).all()
        for t in model.state_dict().values()
        if t.is_floating_point()
    ):
        raise RuntimeError(
            "Checkpoint com pesos não finitos."
        )

    obj["model"] = model
    obj["ema"] = None
    obj["optimizer"] = None

    torch.save(obj, dest)

    return {
        "arquivo": str(dest),
        "sha256": sha256_file(dest),
        "modelo_padrao": True,
        "criterion_serializada": False
    }