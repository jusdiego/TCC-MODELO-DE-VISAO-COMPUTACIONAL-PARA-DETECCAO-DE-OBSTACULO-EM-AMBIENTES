"""Comparação pareada, indícios dos erros e galeria de inspeção."""
from __future__ import annotations

import gzip
import html
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from diag11_core import (NAMES, LEVEL, alert_on, canonical, ratio, save_csv,
                         save_json, with_policy)
from diag11_data import stable_order


def paired_bootstrap(old_images, new_images, repetitions=1000, seed=42):
    old = {r["id"]: r for r in old_images}
    new = {r["id"]: r for r in new_images}
    if set(old) != set(new) or len(old) != len(old_images) or len(new) != len(new_images):
        raise ValueError("Bootstrap exige as mesmas imagens, sem repetições.")
    ids = sorted(old)
    fields = ["tp_alerta", "fn_alerta", "fp_alerta", "alta_sem_alerta", "alta_ref"]
    a = np.array([[old[i][k] for k in fields] for i in ids], dtype=float)
    b = np.array([[new[i][k] for k in fields] for i in ids], dtype=float)

    def scores(x):
        tp, fn, fp, missing, high = x
        return [100*ratio(tp, tp+fp) if tp+fp else np.nan,
                100*ratio(tp, tp+fn) if tp+fn else np.nan,
                100*ratio(2*tp, 2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                fn, fp, missing, 100*ratio(missing, high) if high else np.nan]

    point = np.array(scores(b.sum(axis=0))) - np.array(scores(a.sum(axis=0)))
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(repetitions):
        index = rng.integers(0, len(ids), len(ids))
        results.append(np.array(scores(b[index].sum(axis=0))) -
                       np.array(scores(a[index].sum(axis=0))))
    samples = np.asarray(results)
    labels = ["precision_alertas_pp", "recall_alertas_pp", "f1_alertas_pp",
              "fn_alertas", "fp_alertas", "alta_sem_alerta", "taxa_alta_sem_alerta_pp"]
    rows = []
    for j, name in enumerate(labels):
        values = samples[np.isfinite(samples[:, j]), j]
        low, high = np.quantile(values, [.025, .975]) if len(values) else (None, None)
        rows.append({"metrica": name, "delta_candidato_menos_antigo": float(point[j]) if np.isfinite(point[j]) else None,
                     "ic95_inferior": low, "ic95_superior": high,
                     "reamostragens_validas": len(values), "imagens": len(ids),
                     "metodo": "Bootstrap pareado por imagem; não corrige possível exposição prévia"})
    return rows


def comparison_rows(evaluations):
    rows = []
    for cohort, models in evaluations.items():
        for name, result in models.items():
            s, a = result["summary"], result["summary"]["alertas_por_objeto"]
            rows.append({"coorte": cohort, "modelo": name, "imagens": s["imagens"],
                         "precision_alertas": a["precision"], "recall_alertas": a["recall"],
                         "f1_alertas": a["f1"], "tp_alerta": a["tp"], "fn_alerta": a["fn"],
                         "fp_alerta": a["fp"], "alta_ref": s["alta_ref"],
                         "alta_sem_alerta": s["alta_sem_alerta"],
                         "map50_recorte": s["map50_recorte"],
                         "map50_95_recorte": s["map50_95_recorte"]})
    return rows


def diagnosis_rows(evaluations):
    rows = []
    for cohort, models in evaluations.items():
        for model, result in models.items():
            events = result["events"]
            for cid, name in enumerate(NAMES):
                group = [e for e in events if e["classe_id"] == cid]
                for indication in sorted({x for e in group for x in e["indicios"].split("|") if x}):
                    part = [e for e in group if indication in e["indicios"].split("|")]
                    rows.append({"coorte": cohort, "modelo": model, "classe": name,
                                 "indicio": indication, "eventos": len(part),
                                 "fn_alerta": sum(e["ha_alerta_ref"] and not e["ha_alerta_pred"] for e in part),
                                 "alta_sem_alerta": sum(e["prioridade_ref"] == "alto" and not e["ha_alerta_pred"] for e in part),
                                 "observacao": "Indícios podem se sobrepor; não estabelecem a causa do erro."})
    return rows


def load_prediction_cache(path):
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in records:
                raise ValueError("Cache tem imagem repetida.")
            records[r["id"]] = r["predicoes"]
    return records


def select_gallery_cases(old, new, limit=24, seed=42):
    a, b = {r["id"]: r for r in old}, {r["id"]: r for r in new}
    if set(a) != set(b):
        raise ValueError("Galeria exige os dois modelos nas mesmas imagens.")
    chosen = {}

    def take(label, ordered, number):
        used = 0
        for iid in ordered:
            if iid not in chosen and len(chosen) < limit:
                chosen[iid] = label
                used += 1
                if used == number:
                    break

    worse = sorted(a, key=lambda i: (-(b[i]["alta_sem_alerta"]-a[i]["alta_sem_alerta"]),
                                     -(b[i]["fn_alerta"]-a[i]["fn_alerta"]), stable_order(i, seed)))
    better = sorted(a, key=lambda i: (b[i]["alta_sem_alerta"]-a[i]["alta_sem_alerta"],
                                      b[i]["fn_alerta"]-a[i]["fn_alerta"], stable_order(i, seed)))
    false = sorted(a, key=lambda i: (-(a[i]["fp_alerta"]+b[i]["fp_alerta"]), stable_order(i, seed)))
    take("Candidato com mais omissões", [i for i in worse if
         (b[i]["alta_sem_alerta"], b[i]["fn_alerta"]) >
         (a[i]["alta_sem_alerta"], a[i]["fn_alerta"])], limit//4)
    take("Candidato com menos omissões", [i for i in better if
         (b[i]["alta_sem_alerta"], b[i]["fn_alerta"]) <
         (a[i]["alta_sem_alerta"], a[i]["fn_alerta"])], limit//4)
    take("Falsos alertas avaliáveis", [i for i in false if a[i]["fp_alerta"]+b[i]["fp_alerta"]], limit//4)
    take("Amostra por hash, independente do erro", sorted(a, key=lambda i: stable_order(i, seed)), limit)
    return [{"id": i, "categoria": label} for i, label in chosen.items()]


def font(size):
    candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_panel(record, objects, title):
    canvas = Image.new("RGB", (540, 470), "#101a28")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="#f5f7fa", font=font(18))
    with Image.open(record["image"]) as image:
        image = image.convert("RGB")
        scale = min(520/image.width, 390/image.height)
        image = image.resize((max(1, round(image.width*scale)),
                              max(1, round(image.height*scale))), Image.Resampling.LANCZOS)
    x0, y0 = (540-image.width)//2, 46+(390-image.height)//2
    canvas.paste(image, (x0, y0))
    sx, sy = image.width/record["width"], image.height/record["height"]
    colours = {"alto": "#ff7b8a", "medio": "#ffde70",
               "baixo": "#6ee7b7", "sem_prioridade": "#adbac7"}
    for obj in objects:
        cid = obj["classe_id"]
        known = bool(record["known_mask"] & (1 << cid))
        colour = colours[obj["prioridade"]] if known else "#a9afba"
        x1, y1, x2, y2 = obj["bbox"]
        box = [x0+x1*sx, y0+y1*sy, x0+x2*sx, y0+y2*sy]
        draw.rectangle(box, outline=colour, width=2)
        label = obj["classe"]
        if "confianca" in obj:
            label += f" {obj['confianca']:.2f}"
        label += " / " + obj["prioridade"] if known else " / sem cobertura"
        # Limita o rótulo à largura visível; não altera a caixa nem as métricas.
        tx, ty = max(0, min(box[0], 535)), max(44, box[1]-15)
        text_box = draw.textbbox((tx, ty), label, font=font(11))
        draw.rectangle(text_box, fill="#101a28")
        draw.text((tx, ty), label, fill=colour, font=font(11))
    draw.text((12, 445), "Rosa: alta | amarelo: média | verde: baixa | cinza: sem referência",
              fill="#c6ced9", font=font(10))
    return canvas


def render_case(record, old_pred, new_pred, policy, destination):
    gt = with_policy([g for g in record["gt"] if record["known_mask"] & (1 << g["classe_id"])],
                     record["width"], record["height"], policy)
    old = with_policy([p for p in old_pred if p["confianca"] >= .25],
                      record["width"], record["height"], policy)
    new = with_policy([p for p in new_pred if p["confianca"] >= .25],
                      record["width"], record["height"], policy)
    panels = [draw_panel(record, gt, "Referência V2 avaliável"),
              draw_panel(record, old, "Modelo antigo / confiança 0,25"),
              draw_panel(record, new, "Candidato / confiança 0,25")]
    result = Image.new("RGB", (1620, 470), "#101a28")
    for i, panel in enumerate(panels):
        result.paste(panel, (540*i, 0))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, quality=88)
    return destination


def format_percent(value):
    return "sem suporte" if value is None else f"{100*value:.2f}%"


def build_report(output_dir, evaluations, cohorts, audit, bootstrap, policy, dataset_manifest, seed=42):
    output_dir = Path(output_dir)
    comparison = comparison_rows(evaluations)
    diagnosis = diagnosis_rows(evaluations)
    save_csv(output_dir / "comparacao_modelos.csv", comparison)
    save_csv(output_dir / "diagnostico_erros.csv", diagnosis)
    save_csv(output_dir / "bootstrap_pareado.csv", bootstrap)
    gallery = []
    for cohort, models in evaluations.items():
        records = {r["id"]: r for r in cohorts[cohort]}
        caches = {name: load_prediction_cache(result["cache_file"]) for name, result in models.items()}
        cases = select_gallery_cases(models["antigo"]["images"], models["candidato"]["images"], seed=seed)
        old_images = {r["id"]: r for r in models["antigo"]["images"]}
        new_images = {r["id"]: r for r in models["candidato"]["images"]}
        for i, case in enumerate(cases):
            record = records[case["id"]]
            relative = f"galeria/{cohort}_{i:03}.jpg"
            render_case(record, caches["antigo"][record["id"]],
                        caches["candidato"][record["id"]], policy, output_dir/relative)
            gallery.append({**case, "coorte": cohort, "imagem": record["name"], "arquivo": relative,
                            "antigo": {k: old_images[record["id"]][k] for k in
                                       ["tp_alerta", "fn_alerta", "fp_alerta", "alta_sem_alerta"]},
                            "candidato": {k: new_images[record["id"]][k] for k in
                                          ["tp_alerta", "fn_alerta", "fp_alerta", "alta_sem_alerta"]},
                            "creditos": record.get("creditos", {})})
    save_json(output_dir / "indice_galeria.json", gallery)
    new_records = cohorts["openimages_val_nova"]
    save_csv(output_dir / "revisao_cenarios.csv", [
        {"id": r["id"], "imagem": r["name"], "ambiente_interno": "nao_revisado",
         "proximo_do_uso_pretendido": "nao_revisado", "anotacao_parece_correta": "nao_revisado",
         "observacao": "", "origem": r.get("creditos", {}).get("OriginalLandingURL", "")}
        for r in new_records])
    provenance = audit["baseline_original"]["estado_proveniencia_original"]
    table = "<table><thead><tr><th>Conjunto</th><th>Modelo</th><th>Precisão alertas</th><th>Recall alertas</th><th>F1</th><th>FN</th><th>Alta sem alerta</th></tr></thead><tbody>"
    for r in comparison:
        table += "<tr>" + "".join("<td>"+html.escape(str(x))+"</td>" for x in
                     [r["coorte"], r["modelo"], format_percent(r["precision_alertas"]),
                      format_percent(r["recall_alertas"]), format_percent(r["f1_alertas"]),
                      r["fn_alerta"], r["alta_sem_alerta"]]) + "</tr>"
    table += "</tbody></table>"
    cards = []
    for case in gallery:
        credit = case["creditos"]
        source_url = str(credit.get("OriginalLandingURL", ""))
        license_url = str(credit.get("License", ""))
        attribution = html.escape(str(credit.get("Author", "")))
        if source_url.startswith(("https://", "http://")):
            attribution += ' · <a href="'+html.escape(source_url, quote=True)+'">Fonte original</a>'
        if license_url.startswith(("https://", "http://")):
            attribution += ' · <a href="'+html.escape(license_url, quote=True)+'">Licença</a>'
        details = (f"Antigo: FN {case['antigo']['fn_alerta']}, FP {case['antigo']['fp_alerta']}, "
                   f"alta sem alerta {case['antigo']['alta_sem_alerta']}. "
                   f"Candidato: FN {case['candidato']['fn_alerta']}, FP {case['candidato']['fp_alerta']}, "
                   f"alta sem alerta {case['candidato']['alta_sem_alerta']}.")
        cards.append("<article><h3>"+html.escape(case["coorte"]+" / "+case["categoria"])+
                     "</h3><p>"+html.escape(case["imagem"]+" — "+details)+"</p>"+
                     '<img loading="lazy" src="'+case["arquivo"]+'" alt="Caixas de referência e dos dois modelos">'+
                     "<p class=small>"+attribution+
                     " · Sobreposição de caixas para avaliação.</p></article>")
    html_report = """<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TCC — Diagnóstico dos detectores</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f3f5f7;color:#142135;margin:0}
main{max-width:1400px;margin:auto;padding:32px}h1{font-size:32px;line-height:1.2}
.note,article,table{background:white;border:1px solid #dbe1e8;border-radius:12px}
.note,article{padding:20px;margin:20px 0}table{border-collapse:collapse;width:100%}
td,th{padding:10px;text-align:left;border-bottom:1px solid #e6eaf0}th{background:#e8eff8}
img{max-width:100%;height:auto}.small{font-size:12px;color:#526174}.scroll{overflow:auto}
</style><main><h1>Diagnóstico comparativo dos detectores</h1>
<p>Pesos congelados · 12 classes · política V2 · confiança 0,25 · sem treinamento</p>
<div class=note><strong>Como interpretar</strong><p>
A Home VAL é histórica e exploratória. A validação externa é nova em relação aos IDs
e bytes dos dados conhecidos do projeto; pré-treinamentos públicos não foram auditados.
O recorte Open Images tem cobertura parcial e não garante cenas internas.
As métricas de alertas medem concordância com a V2, não risco físico independente.
A reserva futura não foi inferida. Nenhum modelo foi promovido automaticamente.</p>
<p>Proveniência documental do baseline: <strong>""" + html.escape(provenance) + """</strong>.</p></div>
<h2>Alertas por objeto avaliável</h2><div class=scroll>""" + table + """</div>
<p>Os CSVs trazem suporte por classe, AP no recorte, contagens e intervalos pareados.
Não compare APs de protocolos e distribuições diferentes como se fossem o mesmo benchmark.</p>
<h2>Casos para inspeção</h2><p>Os casos abaixo priorizam diferenças e erros, além de uma amostra por hash.
Esta galeria não é uma estimativa da frequência dos erros. As contagens acima usam todas as imagens.
Objetos sem cobertura aparecem em cinza e não são classificados automaticamente como falsos positivos.</p>
""" + "".join(cards) + """<h2>Próxima decisão</h2><p>
Revisar os cenários e os indícios dos erros; só depois definir o treino de controle.
Os pesos antigos, o candidato e os testes anteriores permanecem preservados.</p></main></html>"""
    (output_dir / "relatorio_diagnostico.html").write_text(html_report, encoding="utf-8")
    lines = ["# Diagnóstico 11", "", "Comparação sem novo treinamento e sem seleção automática.",
             "", f"Proveniência original do baseline: {provenance}.",
             f"Validação externa: {len(new_records)} imagens.",
             f"Reserva futura não inferida: {dataset_manifest['contagens'].get('reserva_futura', 0)} imagens.",
             "", "| Conjunto | Modelo | Precisão alertas | Recall alertas | F1 | FN | Alta sem alerta |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in comparison:
        lines.append("| "+" | ".join(map(str, [r["coorte"], r["modelo"],
                     format_percent(r["precision_alertas"]), format_percent(r["recall_alertas"]),
                     format_percent(r["f1_alertas"]), r["fn_alerta"], r["alta_sem_alerta"]]))+" |")
    lines += ["", "A nova cobertura é conservadora e a seleção busca diversidade de classes; não representa a frequência natural do uso.",
              "A Home VAL continua exploratória enquanto a origem do primeiro treinamento não estiver esclarecida.",
              "Semelhança por pHash é um filtro de suspeitas, não uma prova de identidade de cena.",
              "Os indícios de erro podem se sobrepor e não identificam causas de forma conclusiva.",
              "A reserva atual não foi inferida; não há avaliação independente de risco físico.",
              "", "Veja relatorio_diagnostico.html para a galeria e bootstrap_pareado.csv para a incerteza das diferenças."]
    (output_dir / "relatorio_diagnostico.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return comparison, diagnosis, gallery


def package_reports(output_dir):
    output_dir = Path(output_dir)
    archive = output_dir / "pacote_diagnostico_11.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != archive and not path.name.endswith(".tmp"):
                z.write(path, path.relative_to(output_dir))
    return archive
