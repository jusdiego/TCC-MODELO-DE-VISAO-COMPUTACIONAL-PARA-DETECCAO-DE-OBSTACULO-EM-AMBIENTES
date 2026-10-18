"""Orçamento e seleção do experimento 12, sem dependência de Torch."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import numpy as np

ARMS = {
    "A_home": {"home_per_batch": 16, "oi_per_batch": 0},
    "B_home_replay": {"home_per_batch": 12, "oi_per_batch": 4},
}


class BudgetSampler:
    """Fluxos por fonte independentes; quantidade fixa de batches por época."""
    def __init__(self, home, oi, batch_size, batches, oi_per_batch, seed=42):
        self.home, self.oi = list(home), list(oi)
        self.batch_size, self.batches = int(batch_size), int(batches)
        self.oi_per_batch = int(oi_per_batch)
        self.home_per_batch = self.batch_size - self.oi_per_batch
        self.seed, self.epoch = int(seed), 0
        if (not self.home or self.batches <= 0 or self.batch_size <= 0 or
                not 0 <= self.oi_per_batch < self.batch_size or
                (self.oi_per_batch and not self.oi) or set(self.home) & set(self.oi)):
            raise ValueError("Fontes ou orçamento inválidos.")

    def __len__(self):
        return self.batches*self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _stream(self, values, needed, source):
        if not needed:
            return []
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, source]))
        result = []
        while len(result) < needed:
            result.extend(rng.permutation(values).tolist())
        return result[:needed]

    def __iter__(self):
        home = self._stream(self.home, self.batches*self.home_per_batch, 0)
        oi = self._stream(self.oi, self.batches*self.oi_per_batch, 1)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 2]))
        for b in range(self.batches):
            batch = home[b*self.home_per_batch:(b+1)*self.home_per_batch]
            batch += oi[b*self.oi_per_batch:(b+1)*self.oi_per_batch]
            rng.shuffle(batch)
            yield from batch


def order_id(iid, seed=42):
    return hashlib.sha256(f"{seed}|{iid}".encode()).hexdigest(), iid


def diverse_subset(rows, n=2000, seed=42, per_class=40):
    if n <= 0 or n > len(rows) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Tamanho do replay ou identidade dos IDs inválidos.")
    mapping = {r["id"]: r for r in rows}
    groups = {c: sorted([r["id"] for r in rows if int(r["classe_primaria"]) == c],
                        key=lambda iid: order_id(iid, seed)) for c in range(12)}
    chosen = set()
    for k in range(per_class):
        for c in sorted(groups, key=lambda cid: (len(groups[cid]), cid)):
            if k < len(groups[c]) and len(chosen) < n:
                chosen.add(groups[c][k])
    for iid in sorted(mapping, key=lambda x: order_id(x, seed)):
        if len(chosen) == n:
            break
        chosen.add(iid)
    return [mapping[iid] for iid in sorted(chosen)]


def choose_replay(history_rows, n=2200, seed=42, per_class=40, forbidden_hashes=()):
    batches = {f"lote_{i:06}.zip" for i in range(1, 6)}
    rows = [r for r in history_rows if r["lote"] in batches]
    if len(rows) != 5000 or any(r["split"] != "train" for r in rows):
        raise ValueError("Os cinco lotes do piloto precisam conter 5.000 imagens só de treino.")
    forbidden = set(forbidden_hashes)
    return diverse_subset([r for r in rows if r["imagem_sha256"] not in forbidden],
                          n, seed, per_class)


def validate_metric_block(block, label):
    for k in ["precision", "recall", "f1"]:
        x = block["alertas_por_objeto"][k]
        if x is None or not math.isfinite(x):
            raise ValueError(f"Métrica sem suporte ou não finita: {label}/{k}")
    for k in ["alta_ref", "alta_sem_alerta"]:
        if int(block[k]) < 0:
            raise ValueError("Contagem inválida.")


def make_rule(baseline_home, candidate_external, dataset_sha, policy_sha):
    validate_metric_block(baseline_home, "baseline_home")
    validate_metric_block(candidate_external, "candidate_external")
    return {
        "versao": "12-regra-duas-validacoes-v1",
        "home": {
            "precision_maior_que": baseline_home["alertas_por_objeto"]["precision"],
            "recall_minimo": baseline_home["alertas_por_objeto"]["recall"],
            "alta_sem_alerta_maximo": baseline_home["alta_sem_alerta"],
            "objetos_alerta_ref": baseline_home["alertas_por_objeto"]["tp"]+baseline_home["alertas_por_objeto"]["fn"],
            "alta_ref": baseline_home["alta_ref"],
        },
        "externa": {
            "precision_minima": candidate_external["alertas_por_objeto"]["precision"],
            "recall_minimo": candidate_external["alertas_por_objeto"]["recall"],
            "alta_sem_alerta_maximo": candidate_external["alta_sem_alerta"],
            "objetos_alerta_ref": candidate_external["alertas_por_objeto"]["tp"]+candidate_external["alertas_por_objeto"]["fn"],
            "alta_ref": candidate_external["alta_ref"],
        },
        "dataset_sha256": dataset_sha, "politica_sha256": policy_sha,
        "conf_alertas": .25, "testes_consultados": False,
        "tolerancia_numerica": 1e-12,
        "criterio": "Home supera precisão do baseline, preserva recall e omissões altas; externa preserva precisão, recall e omissões altas do candidato 10.",
        "desempate": "Precisão Home, recall Home, omissões altas Home, F1 externa, menos atualizações, identificador.",
        "sem_elegivel": "Preservar baseline em uso; candidato 10 e todos os checkpoints continuam disponíveis.",
    }


def judge(home, external, rule):
    validate_metric_block(home, "home")
    validate_metric_block(external, "external")
    h, e = home["alertas_por_objeto"], external["alertas_por_objeto"]
    rh, re = rule["home"], rule["externa"]
    for observed, expected in [(h["tp"]+h["fn"], rh["objetos_alerta_ref"]),
                               (e["tp"]+e["fn"], re["objetos_alerta_ref"]),
                               (home["alta_ref"], rh["alta_ref"]),
                               (external["alta_ref"], re["alta_ref"])]:
        if observed != expected:
            raise ValueError("Os denominadores de referência mudaram entre checkpoints.")
    eps = rule["tolerancia_numerica"]
    checks = {
        "home_precision_superior": h["precision"] > rh["precision_maior_que"]+eps,
        "home_recall_preservado": h["recall"] >= rh["recall_minimo"]-eps,
        "home_altas_preservadas": home["alta_sem_alerta"] <= rh["alta_sem_alerta_maximo"],
        "externa_precision_preservada": e["precision"] >= re["precision_minima"]-eps,
        "externa_recall_preservado": e["recall"] >= re["recall_minimo"]-eps,
        "externa_altas_preservadas": external["alta_sem_alerta"] <= re["alta_sem_alerta_maximo"],
    }
    return {**checks, "elegivel": all(checks.values()),
            "retencao_externa": all(v for k, v in checks.items() if k.startswith("externa_"))}


def select_checkpoints(entries, rule):
    if not entries:
        raise ValueError("Nenhum checkpoint avaliado.")
    assessed = []
    for entry in entries:
        checks = judge(entry["home"], entry["external"], rule)
        assessed.append({**entry, "checks": checks})
    eligible = [r for r in assessed if r["checks"]["elegivel"]]

    def ranking(r):
        h = r["home"]["alertas_por_objeto"]
        return (-h["precision"], -h["recall"], r["home"]["alta_sem_alerta"],
                -r["external"]["alertas_por_objeto"]["f1"], r["updates"], r["id"])

    selected = min(eligible, key=ranking) if eligible else None
    # Apenas para inspeção. A classificação exploratória não muda os requisitos.
    retained = [r for r in assessed if r["checks"]["retencao_externa"]]
    exploratory = min(retained, key=lambda r: (
        -r["home"]["alertas_por_objeto"]["recall"],
        r["home"]["alta_sem_alerta"],
        -r["home"]["alertas_por_objeto"]["precision"],
        -r["external"]["alertas_por_objeto"]["f1"],
        r["updates"], r["id"])) if retained else None
    return {"eligible": selected, "exploratory": exploratory, "assessed": assessed}
