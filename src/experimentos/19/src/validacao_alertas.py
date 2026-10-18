"""Compara detectores na validação Home, com a V2 e os parâmetros do 05d."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .politica_alertas import avaliar_bbox

IMGSZ = 640
CONF_ALERTA = 0.25
IOU_NMS = 0.70
IOU_CORRESPONDENCIA = 0.50
MAX_DETECCOES = 300


def ler_objetos_yolo(caminho, largura, altura, nomes):
    objetos = []
    for n, linha in enumerate(Path(caminho).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not linha.strip():
            continue
        partes = linha.split()
        if len(partes) != 5:
            raise ValueError(f"Label inválido: {caminho}, linha {n}")
        valores = [float(p) for p in partes]
        if not all(math.isfinite(v) for v in valores):
            raise ValueError(f"Label não finito: {caminho}, linha {n}")
        id_float, xc, yc, w, h = valores
        c = int(id_float)
        bordas = [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]
        if id_float != c or c not in nomes or w <= 0 or h <= 0 or min(bordas) < -0.0001 or max(bordas) > 1.0001:
            raise ValueError(f"Classe ou caixa inválida: {caminho}, linha {n}")
        x1, y1, x2, y2 = np.clip(bordas, 0, 1)
        objetos.append({"classe_id": c, "classe": nomes[c],
                        "bbox": [float(x1 * largura), float(y1 * altura),
                                 float(x2 * largura), float(y2 * altura)]})
    return objetos


def carregar_home_val(dataset, nomes):
    dataset = Path(dataset)
    raiz_imagens = dataset / "images/val"
    registros = []
    for imagem in sorted(raiz_imagens.rglob("*")):
        if not imagem.is_file() or imagem.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        relativo = imagem.relative_to(raiz_imagens)
        if relativo.parts[0] == "sunrgbd":
            continue
        label = dataset / "labels/val" / relativo.with_suffix(".txt")
        with Image.open(imagem) as im:
            largura, altura = im.size
            im.verify()
        registros.append({"imagem": imagem, "nome": relativo.as_posix(),
                          "largura": largura, "altura": altura,
                          "gt": ler_objetos_yolo(label, largura, altura, nomes)})
    if not registros:
        raise ValueError("Validação Home vazia.")
    return registros


def iou_matriz(gt, pred):
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=float)
    g = np.asarray([o["bbox"] for o in gt], dtype=float)
    p = np.asarray([o["bbox"] for o in pred], dtype=float)
    inicio = np.maximum(g[:, None, :2], p[None, :, :2])
    fim = np.minimum(g[:, None, 2:], p[None, :, 2:])
    inter = np.prod(np.maximum(0.0, fim - inicio), axis=2)
    ag = np.prod(np.maximum(0.0, g[:, 2:] - g[:, :2]), axis=1)
    ap = np.prod(np.maximum(0.0, p[:, 2:] - p[:, :2]), axis=1)
    uniao = ag[:, None] + ap[None, :] - inter
    return np.divide(inter, uniao, out=np.zeros_like(inter), where=uniao > 0)


def corresponder_objetos(gt, pred):
    matriz = iou_matriz(gt, pred)
    candidatos = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["classe_id"] == p["classe_id"] and matriz[gi, pi] >= IOU_CORRESPONDENCIA:
                candidatos.append((float(matriz[gi, pi]), gi, pi))
    candidatos.sort(key=lambda item: (-item[0], item[1], item[2]))
    usados_g, usados_p, pares = set(), set(), []
    for iou, gi, pi in candidatos:
        if gi not in usados_g and pi not in usados_p:
            pares.append((gi, pi, iou))
            usados_g.add(gi)
            usados_p.add(pi)
    return pares, [i for i in range(len(gt)) if i not in usados_g], [i for i in range(len(pred)) if i not in usados_p]


def contagem_eventos(gt, pred, largura, altura):
    """Um FN/FP pode decorrer de detecção ausente/extra ou mudança de prioridade."""
    g = [{**o, **avaliar_bbox(o["classe"], o["bbox"], largura, altura)} for o in gt]
    p = [{**o, **avaliar_bbox(o["classe"], o["bbox"], largura, altura)} for o in pred]
    pares, faltantes, extras = corresponder_objetos(g, p)
    eventos = [(g[gi], p[pi]) for gi, pi, _ in pares]
    eventos += [(g[gi], None) for gi in faltantes]
    eventos += [(None, p[pi]) for pi in extras]
    contagem = {"tp": 0, "fn": 0, "fp": 0, "alta_sem_alerta": 0}
    for ref, prev in eventos:
        a = bool(ref and ref["prioridade"] in {"medio", "alto"})
        b = bool(prev and prev["prioridade"] in {"medio", "alto"})
        contagem["tp"] += int(a and b)
        contagem["fn"] += int(a and not b)
        contagem["fp"] += int(not a and b)
        contagem["alta_sem_alerta"] += int(bool(ref and ref["prioridade"] == "alto" and not b))
    return contagem


def avaliar_alertas(modelo, registros, nomes, device=0):
    modelo_nomes = {int(k): str(v) for k, v in modelo.names.items()}
    if modelo_nomes != nomes:
        raise ValueError("Mapa de classes do modelo difere do projeto.")
    contagem = {"tp": 0, "fn": 0, "fp": 0, "alta_sem_alerta": 0}
    for numero, r in enumerate(registros, 1):
        resultado = modelo.predict(str(r["imagem"]), conf=CONF_ALERTA, imgsz=IMGSZ,
                                   iou=IOU_NMS, max_det=MAX_DETECCOES,
                                   device=device, verbose=False, save=False)[0]
        caixas = resultado.boxes
        pred = [] if caixas is None else [
            {"classe_id": int(c), "classe": nomes[int(c)], "bbox": [float(v) for v in b]}
            for c, b in zip(caixas.cls.cpu().numpy(), caixas.xyxy.cpu().numpy())]
        eventos = contagem_eventos(r["gt"], pred, r["largura"], r["altura"])
        for k, v in eventos.items():
            contagem[k] += v
        if numero % 100 == 0:
            print(f"Validação dos alertas: {numero}/{len(registros)}")
    tp, fn, fp = contagem["tp"], contagem["fn"], contagem["fp"]
    dividir = lambda n, d: n / d if d else None
    return {"imagens": len(registros), **contagem,
            "precision_alertas": dividir(tp, tp + fp),
            "recall_alertas": dividir(tp, tp + fn),
            "f1_alertas": dividir(2 * tp, 2 * tp + fp + fn),
            "referencia": "V2 fixa nas anotações; não valida risco físico independentemente",
            "conf": CONF_ALERTA, "iou_nms": IOU_NMS,
            "iou_correspondencia": IOU_CORRESPONDENCIA}
