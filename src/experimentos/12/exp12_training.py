"""Treino finito e avaliação isolada dos checkpoints."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import math
import random
import hashlib

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.torch_utils import unwrap_model

import diag11_core as core
from exp10_partial import TrainerOpenImages, LoaderFinito, export_standard_checkpoint
from exp12_protocol import BudgetSampler


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(unwrap_model(model).state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


def start_epoch(trainer):
    trainer.train_loader.sampler.set_epoch(trainer.epoch)
    trainer.start_epoch_updates = trainer.actual_updates


def check_epoch(trainer):
    expected = (trainer.epoch+1)*trainer.control_spec["batches"]
    if trainer.actual_updates != expected or trainer.batches_seen != expected:
        raise RuntimeError("Orçamento de atualizações ou batches diferente do planejado.")
    spec = trainer.control_spec
    expected_sources = {"home": expected*(spec["batch"]-spec["oi_per_batch"])}
    if spec["oi_per_batch"]:
        expected_sources["openimages"] = expected*spec["oi_per_batch"]
    if dict(trainer.source_views) != expected_sources:
        raise RuntimeError("Exposição efetiva por fonte diferente da proporção planejada.")


class ControlledTrainer(TrainerOpenImages):
    control_spec = None

    def __init__(self, *args, **kwargs):
        self.actual_updates, self.batches_seen = 0, 0
        self.source_views = Counter()
        super().__init__(*args, **kwargs)
        if self.control_spec is None:
            raise ValueError("Especificação do braço não configurada.")
        self.add_callback("on_train_epoch_start", start_epoch)
        self.add_callback("on_train_epoch_end", check_epoch)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        spec = self.control_spec
        if (batch_size != spec["batch"] or self.args.nbs != spec["batch"]
                or self.args.amp or self.world_size > 1 or self.args.compile
                or self.args.rect or self.args.warmup_epochs != 0):
            raise RuntimeError("Batch, acumulação, precisão ou dispositivo mudou o orçamento fixado.")
        if any(float(getattr(self.args, k, 0)) != 0 for k in
               ["mosaic", "mixup", "cutmix", "copy_paste"]):
            raise RuntimeError("Augmentação composta não é compatível com a máscara por imagem.")
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        home, oi = [], []
        for index, path in enumerate(dataset.im_files):
            key = Path(path).resolve().as_posix()
            if key not in self.source_records:
                raise RuntimeError("Imagem sem proveniência no loader controlado.")
            source = self.source_records[key]["source"]
            if source == "home":
                home.append(index)
            elif source == "openimages":
                oi.append(index)
            else:
                raise RuntimeError("Fonte de treino não reconhecida.")
        if len(home) != spec["home_unique"] or len(oi) != spec["oi_unique"]:
            raise RuntimeError("Contagem de imagens únicas do braço foi alterada.")
        sampler = BudgetSampler(home, oi, batch_size, spec["batches"],
                                spec["oi_per_batch"], self.args.seed)
        return LoaderFinito(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=self.args.workers,
                            pin_memory=self.device.type == "cuda",
                            collate_fn=dataset.collate_fn,
                            generator=torch.Generator().manual_seed(self.args.seed),
                            drop_last=False)

    def preprocess_batch(self, batch):
        if len(batch["im_file"]) != self.control_spec["batch"]:
            raise RuntimeError("Batch incompleto ou redução automática de batch detectada.")
        if self.accumulate != 1:
            raise RuntimeError("A acumulação deixou de ser uma atualização por batch.")
        result = super().preprocess_batch(batch)
        self.batches_seen += 1
        for path in batch["im_file"]:
            self.source_views[self.source_records[Path(path).resolve().as_posix()]["source"]] += 1
        return result

    def optimizer_step(self):
        if not torch.isfinite(self.loss.detach()).all():
            raise FloatingPointError("Loss não finita; o braço foi interrompido.")
        grads = [torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None]
        if grads and not torch.stack(grads).all():
            raise FloatingPointError("Gradiente não finito; o braço foi interrompido.")
        if any(not math.isclose(group["lr"], self.control_spec["lr0"], rel_tol=1e-8, abs_tol=1e-14)
               for group in self.optimizer.param_groups):
            raise RuntimeError("Taxa de aprendizado diferente da taxa constante predefinida.")
        super().optimizer_step()
        self.actual_updates += 1

    def _handle_nan_recovery(self, epoch):
        # A recuperação silenciosa repetiria épocas e invalidaria o orçamento.
        if self.loss is not None and not torch.isfinite(self.loss.detach()).all():
            raise FloatingPointError("Treino não finito; recuperação automática desativada neste controle.")
        if self.fitness is not None and not math.isfinite(float(self.fitness)):
            raise FloatingPointError("Fitness não finito.")
        return False

    def save_model(self):
        for model in [unwrap_model(self.model), self.ema.ema]:
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()
                       if v.is_floating_point()):
                raise FloatingPointError("Peso não finito antes de salvar o checkpoint.")
        return super().save_model()


@contextmanager
def isolated_evaluation():
    """Validação não altera os fluxos de aleatoriedade do treino."""
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    threads = torch.get_num_threads()
    cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    deterministic = (torch.are_deterministic_algorithms_enabled(),
                     torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        torch.set_num_threads(threads)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn
        torch.use_deterministic_algorithms(deterministic[0], warn_only=deterministic[1])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_checkpoint(weight, checkpoint_id, cohorts, output_dir, policy, matcher, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with isolated_evaluation():
        model = YOLO(str(weight))
        if {int(k): str(v) for k, v in model.names.items()} != dict(enumerate(core.NAMES)):
            raise RuntimeError("Classes do checkpoint incompatíveis com as validações.")
        result = {}
        for cohort, records in cohorts.items():
            if any(r["role"] not in {"home_val", "val_nova"} for r in records):
                raise RuntimeError("Reserva ou treino entrou na avaliação de checkpoint.")
            result[cohort] = core.evaluate_model(
                model, records, "candidato", output_dir/cohort, policy, matcher, device, ap_per_class)
        del model
    return result


def metric_entry(checkpoint_id, arm, epoch, updates, weight, evaluated):
    return {"id": checkpoint_id, "arm": arm, "epoch": epoch, "updates": updates,
            "weight": str(weight), "weight_sha256": core.sha256_file(weight),
            "home": evaluated["home_val_historica"]["summary"],
            "external": evaluated["openimages_val_nova"]["summary"]}


def make_checkpoint_callback(arm, cohorts, output_dir, policy, matcher, device,
                             entries, results, backup):
    output_dir = Path(output_dir)
    done = set()

    def after_save(trainer):
        epoch = trainer.epoch+1
        if epoch in done:
            return
        check_epoch(trainer)
        checkpoint_id = f"{arm}_e{epoch:02}"
        weight = output_dir/"pesos_checkpoints"/(checkpoint_id+".pt")
        weight.parent.mkdir(parents=True, exist_ok=True)
        export = export_standard_checkpoint(trainer.last, weight)
        core.save_json(output_dir/"exports"/(checkpoint_id+".json"), export)
        # Preserva o peso antes da avaliação mais demorada.
        backup("checkpoint salvo: "+checkpoint_id)
        evaluated = evaluate_checkpoint(weight, checkpoint_id, cohorts,
                                        output_dir/"avaliacoes"/checkpoint_id,
                                        policy, matcher, device)
        entry = metric_entry(checkpoint_id, arm, epoch, trainer.actual_updates, weight, evaluated)
        entries.append(entry)
        results[checkpoint_id] = evaluated
        core.save_json(output_dir/"checkpoints_avaliados.json", entries)
        core.save_json(output_dir/"orcamento"/(checkpoint_id+".json"),
                       {"arm": arm, "epoch": epoch, "updates": trainer.actual_updates,
                        "batches_seen": trainer.batches_seen, "source_views": dict(trainer.source_views),
                        "batch": trainer.control_spec["batch"], "amp": False, "accumulate": trainer.accumulate})
        done.add(epoch)
        backup("checkpoint avaliado: "+checkpoint_id)
    return after_save
