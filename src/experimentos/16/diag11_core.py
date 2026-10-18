"""Núcleo do diagnóstico 11. Não contém treinamento nem seleção de pesos."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

NAMES = ["bed", "sofa", "chair", "table", "lamp", "tv", "laptop",
         "wardrobe", "window", "door", "potted plant", "photo frame"]
ALL_KNOWN = (1 << len(NAMES)) - 1
ALERT_IDS = {0, 1, 2, 3, 7, 10}
ALERT_MASK = sum(1 << c for c in ALERT_IDS)
LEVEL = {"sem_prioridade": 0, "baixo": 1, "medio": 2, "alto": 3}


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_clean(value), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"Membro inseguro no ZIP: {name}")
    return p


def parse_yolo(raw, width, height, names=NAMES, source="label"):
    objects = []
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{source}:{line_no}: esperados cinco campos.")
        vals = [float(s) for s in parts]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"{source}:{line_no}: valor não finito.")
        c, x, y, w, h = vals
        if int(c) != c or not 0 <= c < len(names) or min(w, h) <= 0:
            raise ValueError(f"{source}:{line_no}: classe ou dimensão inválida.")
        box = [x-w/2, y-h/2, x+w/2, y+h/2]
        if min(box) < -0.002 or max(box) > 1.002:
            raise ValueError(f"{source}:{line_no}: caixa fora da imagem.")
        box = [max(0, min(1, v)) for v in box]
        objects.append({"classe_id": int(c), "classe": names[int(c)],
                        "bbox": [box[0]*width, box[1]*height,
                                 box[2]*width, box[3]*height],
                        "ocluido": None, "truncado": None})
    return objects


def hierarchy_mapping(cfg, class_rows, hierarchy):
    if cfg["nomes"] != NAMES or len(cfg["mids_raizes"]) != len(NAMES):
        raise ValueError("Classes da preparação incompatíveis com o projeto.")
    descriptions = {r[0]: r[1] for r in class_rows if len(r) >= 2}
    roots = cfg["mids_raizes"]
    if any(mid not in descriptions for mid in roots):
        raise ValueError("A descrição oficial não contém todas as raízes.")
    descendants, ancestors = defaultdict(set), defaultdict(set)

    def walk(node, parents=()):
        mid = node["LabelName"]
        ancestors[mid].update(parents)
        descendants[mid].add(mid)
        for parent in parents:
            descendants[parent].add(mid)
        for child in node.get("Subcategory", []):
            walk(child, parents + (mid,))

    walk(hierarchy)
    excluded_names = set(cfg.get("ambiguas_excluidas", []))
    excluded = {mid for mid, name in descriptions.items() if name in excluded_names}
    candidates = defaultdict(set)
    root_desc = {}
    for cid, mid in enumerate(roots):
        if mid not in descendants:
            raise ValueError(f"Raiz ausente da hierarquia: {mid}")
        root_desc[cid] = set(descendants[mid])
        for sub in root_desc[cid]:
            candidates[sub].add(cid)
    ambiguous = excluded | {mid for mid, ids in candidates.items() if len(ids) > 1}
    mapping = {mid: next(iter(ids)) for mid, ids in candidates.items()
               if mid not in ambiguous}
    return {"roots": roots, "descendants": root_desc, "ancestors": ancestors,
            "mapping": mapping, "ambiguous": ambiguous,
            "descriptions": descriptions,
            "relevant_labels": set().union(*root_desc.values(),
                                            *(ancestors[r] for r in roots))}


def prepare_oi_annotations(box_rows, label_rows, info):
    """Cobertura conservadora: positivo na raiz ou negativo na raiz/ancestral.

    Uma positiva apenas numa subclasse não prova cobertura exaustiva da classe
    agregada. Grupos, representações, atributos desconhecidos ou caixas inválidas
    tornam essa classe/imagem não avaliável, em vez de criarem falsos negativos.
    """
    labels = defaultdict(lambda: {"positive": set(), "negative": set()})
    for row in label_rows:
        if row.get("Source") not in {"verification", "crowdsource-verification"}:
            continue
        mid, confidence = row["LabelName"], float(row["Confidence"])
        if mid not in info["relevant_labels"]:
            continue
        if confidence in (0, 1):
            key = "positive" if confidence == 1 else "negative"
            labels[row["ImageID"]][key].add(mid)
    boxes, blocked = defaultdict(list), defaultdict(lambda: defaultdict(set))
    audited = Counter()
    for row in box_rows:
        mid, iid = row["LabelName"], row["ImageID"]
        affected = [cid for cid, mids in info["descendants"].items() if mid in mids]
        if not affected:
            continue
        if mid in info["ambiguous"]:
            for cid in affected:
                blocked[iid][cid].add("subclasse_ambigua_excluida")
            audited["caixas_subclasse_ambigua"] += 1
            continue
        cid = info["mapping"].get(mid)
        if cid is None:
            continue
        flags = {k: int(float(row.get(k, -1))) for k in
                 ["IsGroupOf", "IsDepiction", "IsOccluded", "IsTruncated"]}
        reasons = []
        if row.get("Source") not in {"xclick", "activemil"}:
            reasons.append("origem_caixa_nao_reconhecida")
        if flags["IsGroupOf"] != 0:
            reasons.append("grupo_ou_atributo_desconhecido")
        if flags["IsDepiction"] != 0:
            reasons.append("representacao_ou_atributo_desconhecido")
        xyxy = [float(row[k]) for k in ["XMin", "YMin", "XMax", "YMax"]]
        if (not all(math.isfinite(v) for v in xyxy) or min(xyxy) < 0
                or max(xyxy) > 1 or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]):
            reasons.append("coordenadas_invalidas")
        if reasons:
            blocked[iid][cid].update(reasons)
            audited.update(reasons)
            continue
        boxes[iid].append({"classe_id": cid, "classe": NAMES[cid],
                           "bbox_norm": xyxy, "mid_original": mid,
                           "ocluido": flags["IsOccluded"] == 1 if flags["IsOccluded"] >= 0 else None,
                           "truncado": flags["IsTruncated"] == 1 if flags["IsTruncated"] >= 0 else None})
    records = {}
    for iid in sorted(set(labels) | set(boxes)):
        gt = []
        seen = set()
        for obj in boxes[iid]:
            key = (obj["classe_id"], *(round(x, 8) for x in obj["bbox_norm"]))
            if key not in seen:
                seen.add(key)
                gt.append(obj)
        known, positive, negative, reasons = 0, 0, 0, {}
        present = {o["classe_id"] for o in gt}
        for cid, root in enumerate(info["roots"]):
            pos = root in labels[iid]["positive"]
            neg = bool(({root} | info["ancestors"][root]) & labels[iid]["negative"])
            has_positive_label = bool(info["descendants"][cid] & labels[iid]["positive"])
            reason = set(blocked[iid].get(cid, set()))
            if neg and (cid in present or has_positive_label):
                reason.add("contradicao_positivo_negativo")
            if pos and cid not in present:
                reason.add("positivo_sem_caixa_utilizavel")
            if reason:
                reasons[str(cid)] = sorted(reason)
                continue
            if neg:
                known |= 1 << cid
                negative |= 1 << cid
            elif pos and cid in present:
                known |= 1 << cid
                positive |= 1 << cid
        if positive or known & ALERT_MASK == ALERT_MASK:
            records[iid] = {"id": iid, "source_split": "validation",
                            "known_mask": known, "positive_mask": positive,
                            "negative_mask": negative, "gt_normalizado": gt,
                            "classes_positivas_avaliaveis": sorted(c for c in range(12)
                                                                 if positive & (1 << c)),
                            "restricoes_cobertura": reasons}
    return records, dict(audited)


def iou_matrix(gt, pred):
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    a = np.asarray([o["bbox"] for o in gt], dtype=float)
    b = np.asarray([o["bbox"] for o in pred], dtype=float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(rb-lt, 0).prod(axis=2)
    aa = np.maximum(a[:, 2:]-a[:, :2], 0).prod(axis=1)
    ba = np.maximum(b[:, 2:]-b[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(aa[:, None] + ba[None, :] - intersection, 1e-12)


def ap_matches(gt, pred, thresholds=np.linspace(.5, .95, 10)):
    correct = np.zeros((len(pred), len(thresholds)), dtype=bool)
    matrix = iou_matrix(gt, pred)
    order = sorted(range(len(pred)), key=lambda pi: (-pred[pi]["confianca"], pi))
    for ti, threshold in enumerate(thresholds):
        used = set()
        for pi in order:
            options = [gi for gi, g in enumerate(gt) if gi not in used
                       and g["classe_id"] == pred[pi]["classe_id"]
                       and matrix[gi, pi] >= threshold]
            if options:
                gi = max(options, key=lambda k: (matrix[k, pi], -k))
                used.add(gi)
                correct[pi, ti] = True
    return correct


def ratio(n, d):
    return n/d if d else None


def alert_metrics(tp, fn, fp):
    return {"tp": int(tp), "fn": int(fn), "fp": int(fp),
            "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
            "f1": ratio(2*tp, 2*tp+fp+fn)}


def alert_on(obj):
    return bool(obj and LEVEL[obj["prioridade"]] >= LEVEL["medio"])


def with_policy(objects, width, height, policy):
    return [{**o, **policy.avaliar_bbox(o["classe"], o["bbox"], width, height,
                                       versao="v2")} for o in objects]


def record_event(record, model_name, event, g=None, p=None, iou=None, all_pred=()):
    gbox = g["bbox"] if g else None
    pbox = p["bbox"] if p else None
    area = ((gbox[2]-gbox[0])*(gbox[3]-gbox[1]) /
            (record["width"]*record["height"])) if gbox else None
    side_at_640 = (min(gbox[2]-gbox[0], gbox[3]-gbox[1]) *
                   640/max(record["width"], record["height"])) if gbox else None
    flags = []
    near_best = None
    if event == "objeto_nao_detectado":
        if side_at_640 < 32:
            flags.append("objeto_pequeno_na_entrada_640")
        if g.get("ocluido") is True:
            flags.append("oclusao_anotada")
        if g.get("truncado") is True:
            flags.append("truncamento_anotado")
        overlaps = iou_matrix([g], all_pred)[0] if all_pred else []
        same = [i for i, o in enumerate(all_pred) if o["classe_id"] == g["classe_id"]]
        if same:
            best = max(same, key=lambda i: overlaps[i])
            near_best = float(overlaps[best])
            if overlaps[best] >= .5 and all_pred[best]["confianca"] < .25:
                flags.append("deteccao_abaixo_da_confianca_025")
            elif .1 <= overlaps[best] < .5:
                flags.append("caixa_mesma_classe_iou_010_a_050")
        if any(overlaps[i] >= .5 and o["classe_id"] != g["classe_id"]
               and o["confianca"] >= .25 for i, o in enumerate(all_pred)):
            flags.append("sobreposicao_com_previsao_de_outra_classe")
        if not flags:
            flags.append("omissao_sem_indicio_automatico")
    if event == "previsao_nao_associada":
        flags.append("previsao_sem_correspondencia_na_classe_avaliavel")
    return {"coorte": record["coorte"], "imagem": record["name"], "id": record["id"],
            "modelo": model_name, "evento": event,
            "classe_id": g["classe_id"] if g else p["classe_id"],
            "classe": g["classe"] if g else p["classe"],
            "bbox_ref": json.dumps(gbox) if g else "",
            "bbox_pred": json.dumps(pbox) if p else "",
            "iou": iou, "confianca": p["confianca"] if p else None,
            "prioridade_ref": g["prioridade"] if g else "sem_prioridade",
            "prioridade_pred": p["prioridade"] if p else "sem_prioridade",
            "ha_alerta_ref": alert_on(g), "ha_alerta_pred": alert_on(p),
            "area_relativa_gt": area, "menor_lado_gt_em_640": side_at_640,
            "oclusao_anotada": g.get("ocluido") if g else None,
            "truncamento_anotado": g.get("truncado") if g else None,
            "melhor_iou_mesma_classe": near_best,
            "indicios": "|".join(flags)}


def analyse_prediction(record, pred_low, model_name, policy, matcher):
    if record.get("role") not in {"home_val", "val_nova"}:
        raise ValueError("Somente validação pode entrar na inferência do notebook 11.")
    known = int(record["known_mask"])
    gt = with_policy([g for g in record["gt"] if known & (1 << g["classe_id"])],
                     record["width"], record["height"], policy)
    pred_low_known = [p for p in pred_low if known & (1 << p["classe_id"])]
    pred_all = with_policy([p for p in pred_low if p["confianca"] >= .25],
                           record["width"], record["height"], policy)
    pred = [p for p in pred_all if known & (1 << p["classe_id"])]
    pairs, missing, extras = matcher(gt, pred)
    events = []
    for gi, pi, iou in pairs:
        events.append(record_event(record, model_name, "correspondencia",
                                   gt[gi], pred[pi], float(iou)))
    for gi in missing:
        events.append(record_event(record, model_name, "objeto_nao_detectado",
                                   g=gt[gi], all_pred=pred_low_known))
    for pi in extras:
        events.append(record_event(record, model_name, "previsao_nao_associada", p=pred[pi]))
    tp = sum(e["ha_alerta_ref"] and e["ha_alerta_pred"] for e in events)
    fn = sum(e["ha_alerta_ref"] and not e["ha_alerta_pred"] for e in events)
    fp = sum(not e["ha_alerta_ref"] and e["ha_alerta_pred"] for e in events)
    high_n = sum(g["prioridade"] == "alto" for g in gt)
    high_missing = sum(e["prioridade_ref"] == "alto" and not e["ha_alerta_pred"] for e in events)
    messages = [policy.gerar_mensagem(p, "medio") for p in pred_all if alert_on(p)]
    if any(not isinstance(m, str) or not m.strip() for m in messages):
        raise RuntimeError("A política produziu alerta sem texto.")
    image = {"coorte": record["coorte"], "imagem": record["name"], "id": record["id"],
             "modelo": model_name, "tp_alerta": tp, "fn_alerta": fn, "fp_alerta": fp,
             "alta_ref": high_n, "alta_sem_alerta": high_missing,
             "tp_detector": len(pairs), "fn_detector": len(missing), "fp_detector": len(extras),
             "gt_avaliaveis": len(gt), "gt_ignorados": len(record["gt"])-len(gt),
             "predicoes_ignoradas_conf_025": len(pred_all)-len(pred),
             "cobertura_completa_alertas": known & ALERT_MASK == ALERT_MASK,
             "ha_alerta_ref": any(alert_on(g) for g in gt),
             "ha_alerta_pred": any(alert_on(p) for p in pred),
             "mensagem_alerta": "\n".join(messages)}
    stats = (ap_matches(gt, pred_low_known),
             np.array([p["confianca"] for p in pred_low_known], dtype=float),
             np.array([p["classe_id"] for p in pred_low_known], dtype=int),
             np.array([g["classe_id"] for g in gt], dtype=int))
    return image, events, stats


def summarise(images, events, stats, ap_function=None):
    if not images:
        raise ValueError("Coorte de avaliação vazia.")
    total = lambda k: sum(int(r[k]) for r in images)
    output = {"imagens": len(images), "conf_alertas": .25,
              "alertas_por_objeto": alert_metrics(total("tp_alerta"), total("fn_alerta"), total("fp_alerta")),
              "alta_ref": total("alta_ref"), "alta_sem_alerta": total("alta_sem_alerta"),
              "taxa_alta_sem_alerta": ratio(total("alta_sem_alerta"), total("alta_ref")),
              "detector_conf_025": alert_metrics(total("tp_detector"), total("fn_detector"), total("fp_detector")),
              "gt_ignorados_sem_cobertura": total("gt_ignorados"),
              "predicoes_ignoradas_conf_025": total("predicoes_ignoradas_conf_025")}
    complete = [r for r in images if r["cobertura_completa_alertas"]]
    output["imagens_com_cobertura_completa_alertas"] = len(complete)
    output["alertas_por_imagem_cobertura_completa"] = alert_metrics(
        sum(r["ha_alerta_ref"] and r["ha_alerta_pred"] for r in complete),
        sum(r["ha_alerta_ref"] and not r["ha_alerta_pred"] for r in complete),
        sum(not r["ha_alerta_ref"] and r["ha_alerta_pred"] for r in complete))
    correct = np.concatenate([s[0] for s in stats], axis=0)
    scores = np.concatenate([s[1] for s in stats])
    pred_cls = np.concatenate([s[2] for s in stats])
    target_cls = np.concatenate([s[3] for s in stats])
    ap = {}
    if ap_function is not None and len(target_cls) and len(pred_cls):
        result = ap_function(correct, scores, pred_cls, target_cls, plot=False,
                             names=dict(enumerate(NAMES)))
        values, ids = np.asarray(result[5]), np.asarray(result[6], dtype=int)
        if values.shape != (len(ids), 10):
            raise RuntimeError("Retorno de AP incompatível com a versão fixada.")
        ap = {int(c): a for c, a in zip(ids, values)}
    elif ap_function is not None and len(target_cls):
        ap = {int(c): np.zeros(10) for c in np.unique(target_cls)}
    output["map50_recorte"] = float(np.mean([a[0] for a in ap.values()])) if ap else None
    output["map50_95_recorte"] = float(np.mean(list(ap.values()))) if ap else None
    class_rows = []
    for cid, name in enumerate(NAMES):
        part = [e for e in events if e["classe_id"] == cid]
        a = ap.get(cid)
        n_gt = int((target_cls == cid).sum())
        class_rows.append({"classe_id": cid, "classe": name, "gt_avaliaveis": n_gt,
                           "ap50": float(a[0]) if a is not None else None,
                           "ap50_95": float(a.mean()) if a is not None else None,
                           "suporte_reduzido": n_gt < 30,
                           **alert_metrics(sum(e["ha_alerta_ref"] and e["ha_alerta_pred"] for e in part),
                                           sum(e["ha_alerta_ref"] and not e["ha_alerta_pred"] for e in part),
                                           sum(not e["ha_alerta_ref"] and e["ha_alerta_pred"] for e in part)),
                           "alta_ref": sum(e["prioridade_ref"] == "alto" for e in part),
                           "alta_sem_alerta": sum(e["prioridade_ref"] == "alto" and not e["ha_alerta_pred"] for e in part)})
    return output, class_rows


def model_predictions(model, record, device):
    if record["role"] not in {"home_val", "val_nova"}:
        raise ValueError("Tentativa de inferência em reserva ou conjunto não autorizado.")
    result = model.predict(str(record["image"]), conf=.001, imgsz=640, rect=True,
                           iou=.70, max_det=300, device=device, half=False,
                           verbose=False, save=False)[0]
    if Path(result.path).resolve() != Path(record["image"]).resolve():
        raise RuntimeError("A inferência retornou uma imagem diferente.")
    if result.boxes is None:
        return []
    return [{"classe_id": int(c), "classe": NAMES[int(c)],
             "bbox": [float(v) for v in box], "confianca": float(score)}
            for c, box, score in zip(result.boxes.cls.cpu().numpy(),
                                      result.boxes.xyxy.cpu().numpy(),
                                      result.boxes.conf.cpu().numpy())]


def evaluate_model(model, records, model_name, output_dir, policy, matcher, device,
                   ap_function):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images, events, stats = [], [], []
    cache_file = output_dir / "predicoes_conf_0001.jsonl.gz"
    with gzip.open(cache_file, "wt", encoding="utf-8") as cache:
        for i, record in enumerate(records, 1):
            pred = model_predictions(model, record, device)
            cache.write(json.dumps({"id": record["id"], "imagem": record["name"],
                                    "predicoes": pred}, ensure_ascii=False) + "\n")
            im, ev, st = analyse_prediction(record, pred, model_name, policy, matcher)
            images.append(im)
            events.extend(ev)
            stats.append(st)
            if i % 100 == 0 or i == len(records):
                print(f"{record['coorte']} / {model_name}: {i}/{len(records)}", flush=True)
    summary, classes = summarise(images, events, stats, ap_function)
    summary.update(modelo=model_name, coorte=records[0]["coorte"],
                   protocolo="12 classes; AP por confiança; alertas por IoU; cobertura conservadora",
                   nao_e_openimages_challenge=True, selecao_automatica_de_modelo=False)
    save_json(output_dir / "metricas.json", summary)
    save_csv(output_dir / "metricas_por_classe.csv", classes)
    save_csv(output_dir / "alertas_por_imagem.csv", images)
    save_csv(output_dir / "avaliacao_objetos.csv", events)
    (output_dir / "alertas_texto.txt").write_text(
        "\n\n".join(r["imagem"]+"\n"+r["mensagem_alerta"] for r in images if r["mensagem_alerta"]),
        encoding="utf-8")
    return {"summary": summary, "classes": classes, "images": images, "events": events,
            "cache_file": cache_file}
