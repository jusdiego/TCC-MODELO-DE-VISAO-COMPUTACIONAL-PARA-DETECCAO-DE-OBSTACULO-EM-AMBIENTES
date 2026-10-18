"""Preparação automática por grupos, com proteção do TEST já existente."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .auditoria_dados import CLASSES_HOMEOBJECTS, auditar_dataset, validar_resultados_auditoria


def identificador_execucao() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def salvar_json(caminho, conteudo):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def ler_classes(dataset):
    dados = yaml.safe_load((Path(dataset) / "data.yaml").read_text(encoding="utf-8-sig"))
    nomes = dados["names"]
    return {i: str(v) for i, v in enumerate(nomes)} if isinstance(nomes, list) else {int(i): str(v) for i, v in nomes.items()}


def resumo_test(manifesto, classes, esperado=403):
    registros = manifesto.loc[manifesto["split"].eq("test")].sort_values("relativo")
    if len(registros) != esperado:
        raise RuntimeError(f"Esperado TEST com {esperado} imagens; encontrado {len(registros)}.")
    arquivos = registros[["relativo", "sha256", "label_sha256"]].to_dict("records")
    payload = {"classes": classes, "arquivos": arquivos}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"imagens": len(registros), "sha256": digest, **payload}


def proteger_dataset_existente(dataset, pasta_relatorios, classes=None, esperado=403):
    """Audita a base existente e cria/confere uma referência persistente do TEST."""
    dataset, pasta_relatorios = Path(dataset), Path(pasta_relatorios)
    classes = classes or CLASSES_HOMEOBJECTS
    if ler_classes(dataset) != classes:
        raise RuntimeError("O mapa de classes do dataset não corresponde ao projeto.")
    resultados = auditar_dataset(dataset, pasta_relatorios / "auditoria_base_existente",
                                 classes=classes, gerar_figuras=False)
    validar_resultados_auditoria(resultados)
    snapshot = resumo_test(resultados["manifesto"], classes, esperado)
    raiz = dataset
    while raiz.parent != raiz and not (raiz / "src").is_dir():
        raiz = raiz.parent
    if not (raiz / "src").is_dir():
        raise FileNotFoundError("Não encontrei a pasta src para registrar a proteção do TEST.")
    trava = raiz / "reports/protecao_test/test_congelado.json"
    if trava.exists():
        anterior = json.loads(trava.read_text(encoding="utf-8"))
        if anterior["sha256"] != snapshot["sha256"]:
            raise RuntimeError("O TEST difere da referência congelada. A execução foi bloqueada; não sobrescreva a referência para ocultar a alteração.")
    else:
        salvar_json(trava, snapshot)
    salvar_json(pasta_relatorios / "test_conferido.json", snapshot)
    return resultados, snapshot


class _Uniao:
    def __init__(self, n):
        self.pais = list(range(n))

    def raiz(self, i):
        while i != self.pais[i]:
            self.pais[i] = self.pais[self.pais[i]]
            i = self.pais[i]
        return i

    def juntar(self, a, b):
        a, b = self.raiz(a), self.raiz(b)
        if a != b:
            self.pais[max(a, b)] = min(a, b)


def _anotacao_canonica(caminho):
    linhas = []
    for linha in Path(caminho).read_text(encoding="utf-8-sig").splitlines():
        if linha.strip():
            partes = linha.split()
            coordenadas = [f"{float(v):.12g}" for v in partes[1:]]
            linhas.append(" ".join([str(int(float(partes[0]))), *coordenadas]))
    return "\n".join(sorted(linhas))


def agrupar_desenvolvimento(manifesto_raw, manifesto_test):
    """dHash cria grupos conservadores; só imagem+anotação iguais são deduplicadas."""
    raw = manifesto_raw.sort_values(["split", "relativo"]).reset_index(drop=True).copy()
    uniao = _Uniao(len(raw))
    for coluna in ["sha256", "dhash"]:
        primeiro = {}
        for i, valor in enumerate(raw[coluna]):
            if pd.notna(valor):
                if valor in primeiro:
                    uniao.juntar(i, primeiro[valor])
                else:
                    primeiro[valor] = i
    raw["grupo"] = [f"g{uniao.raiz(i):06d}" for i in range(len(raw))]
    sha_test = set(manifesto_test["sha256"])
    dhash_test = set(manifesto_test["dhash"])
    tocando_test = raw["sha256"].isin(sha_test) | raw["dhash"].isin(dhash_test)
    grupos_test = set(raw.loc[tocando_test, "grupo"])
    exclusoes, mantidas, vistos = [], [], {}
    for i, linha in raw.iterrows():
        if linha["grupo"] in grupos_test:
            exclusoes.append({"split_original": linha["split"], "relativo": linha["relativo"],
                              "grupo": linha["grupo"], "motivo": "grupo_relacionado_ao_test_congelado"})
            continue
        # Mesma imagem com anotações diferentes permanece no mesmo grupo.
        chave = (linha["sha256"], _anotacao_canonica(linha["label"]))
        if chave in vistos:
            exclusoes.append({"split_original": linha["split"], "relativo": linha["relativo"],
                              "grupo": linha["grupo"], "motivo": "imagem_e_anotacao_identicas",
                              "representante": vistos[chave]})
        else:
            vistos[chave] = f"{linha['split']}/{linha['relativo']}"
            mantidas.append(i)
    desenvolvimento = raw.loc[mantidas].copy().reset_index(drop=True)
    desenvolvimento["ids_classes"] = desenvolvimento["ids_classes"].map(lambda s: frozenset(json.loads(s)))
    return desenvolvimento, pd.DataFrame(exclusoes, columns=["split_original", "relativo", "grupo", "motivo", "representante"])


def dividir_por_grupos(desenvolvimento, classes, proporcao=0.20, semente=42):
    """Estratificação heurística por grupos, sem quebrar grupos perceptuais."""
    if not 0 < proporcao < 1 or desenvolvimento.empty:
        raise ValueError("Proporção inválida ou desenvolvimento vazio.")
    grupos = {}
    for grupo, registros in desenvolvimento.groupby("grupo", sort=True):
        contagem = Counter(c for ids in registros["ids_classes"] for c in ids)
        grupos[grupo] = {"n": len(registros), "classes": contagem}
    totais = Counter(c for ids in desenvolvimento["ids_classes"] for c in ids)
    for c in classes:
        suporte = sum(c in g["classes"] for g in grupos.values())
        if suporte < 2:
            raise RuntimeError(f"Classe {classes[c]} aparece em {suporte} grupo(s). Não pode estar em train e val preservando os grupos; reforce a base de desenvolvimento.")
    alvo_val = round(len(desenvolvimento) * proporcao)
    alvos_n = {"train": len(desenvolvimento) - alvo_val, "val": alvo_val}
    alvos_c = {s: {c: totais[c] * (1 - proporcao if s == "train" else proporcao) for c in classes}
               for s in ["train", "val"]}

    melhor = None
    for tentativa in range(32):
        rng = random.Random(semente + tentativa)
        desempates = {g: rng.random() for g in grupos}
        ordem = sorted(grupos, key=lambda g: (min((totais[c] for c in grupos[g]["classes"]), default=10**12),
                                              desempates[g], g))
        n = {"train": 0, "val": 0}
        contagens = {"train": Counter(), "val": Counter()}
        atribuicoes = {}
        for grupo in ordem:
            info = grupos[grupo]
            custos = {}
            for split in ["train", "val"]:
                antes = (n[split] - alvos_n[split]) ** 2 / max(alvos_n[split], 1)
                depois = (n[split] + info["n"] - alvos_n[split]) ** 2 / max(alvos_n[split], 1)
                custo = 2 * (depois - antes)
                for c, qtd in info["classes"].items():
                    alvo = max(alvos_c[split][c], 1)
                    custo += ((contagens[split][c] + qtd - alvo) ** 2
                              - (contagens[split][c] - alvo) ** 2) / alvo
                custos[split] = custo
            escolhido = min(["train", "val"], key=lambda s: (custos[s], n[s] / max(alvos_n[s], 1)))
            atribuicoes[grupo] = escolhido
            n[escolhido] += info["n"]
            contagens[escolhido].update(info["classes"])

        # Corrige falta de classe somente com movimentos que preservam todas as demais.
        for destino in ["train", "val"]:
            origem = "val" if destino == "train" else "train"
            for c in sorted(classes, key=lambda c: (totais[c], c)):
                if contagens[destino][c]:
                    continue
                candidatos = [g for g in grupos if atribuicoes[g] == origem and c in grupos[g]["classes"]
                              and all(contagens[origem][k] - qtd > 0 for k, qtd in grupos[g]["classes"].items())]
                if candidatos:
                    g = min(candidatos, key=lambda g: (abs(n[destino] + grupos[g]["n"] - alvos_n[destino]), g))
                    atribuicoes[g] = destino
                    n[destino] += grupos[g]["n"]
                    n[origem] -= grupos[g]["n"]
                    contagens[destino].update(grupos[g]["classes"])
                    contagens[origem].subtract(grupos[g]["classes"])
        if any(contagens[s][c] == 0 for s in ["train", "val"] for c in classes):
            continue
        custo = abs(n["val"] - alvo_val) / len(desenvolvimento)
        custo += sum(abs(contagens["val"][c] / totais[c] - proporcao) for c in classes) / len(classes)
        if melhor is None or custo < melhor[0]:
            melhor = custo, atribuicoes.copy()
    if melhor is None:
        raise RuntimeError("Não foi encontrada uma divisão válida por grupos. Reforce as classes raras; os grupos e o TEST não foram alterados.")
    resultado = desenvolvimento.copy()
    resultado["split_final"] = resultado["grupo"].map(melhor[1])
    if (resultado.groupby("grupo")["split_final"].nunique() > 1).any():
        raise AssertionError("Grupo dividido entre train e val.")
    return resultado


def criar_dataset_automatico(raiz_raw, dataset_congelado, raiz_projeto, classes=None, proporcao=0.20,
                            semente=42, esperado_test=403):
    raiz_projeto, raiz_raw = Path(raiz_projeto), Path(raiz_raw)
    classes = classes or CLASSES_HOMEOBJECTS
    identificador = identificador_execucao()
    relatorios = raiz_projeto / "reports/preparacao_automatica" / identificador
    congelado, referencia = proteger_dataset_existente(dataset_congelado, relatorios, classes, esperado_test)
    auditoria = auditar_dataset(raiz_raw, relatorios / "auditoria_raw", classes, gerar_figuras=False)
    validar_resultados_auditoria(auditoria)
    test = congelado["manifesto"].loc[congelado["manifesto"]["split"].eq("test")].copy()
    desenvolvimento, exclusoes = agrupar_desenvolvimento(auditoria["manifesto"], test)
    final = dividir_por_grupos(desenvolvimento, classes, proporcao, semente)

    saida = raiz_projeto / "data/processed/homeobjects_yolo_automatico" / identificador
    saida.mkdir(parents=True, exist_ok=False)
    for _, linha in final.iterrows():
        # Namespace de origem evita sobrescrever nomes iguais de splits diferentes.
        relativo = Path("raw_" + linha["split"]) / Path(linha["relativo"])
        for tipo, fonte, destino_rel in [("images", linha["imagem"], relativo),
                                        ("labels", linha["label"], relativo.with_suffix(".txt"))]:
            destino = saida / tipo / linha["split_final"] / destino_rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fonte, destino)
    for _, linha in test.iterrows():
        relativo = Path(linha["relativo"])
        for tipo, fonte, destino_rel in [("images", linha["imagem"], relativo),
                                        ("labels", linha["label"], relativo.with_suffix(".txt"))]:
            destino = saida / tipo / "test" / destino_rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fonte, destino)
    (saida / "data.yaml").write_text(yaml.safe_dump({"path": saida.as_posix(), "train": "images/train",
        "val": "images/val", "test": "images/test", "names": classes}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    verificada = auditar_dataset(saida, relatorios / "auditoria_final", classes, gerar_figuras=False)
    validar_resultados_auditoria(verificada)
    if resumo_test(verificada["manifesto"], classes, esperado_test)["sha256"] != referencia["sha256"]:
        raise RuntimeError("A cópia do TEST não corresponde à referência.")
    # Verifica os bytes do desenvolvimento copiado contra a auditoria da origem.
    esperados = {(linha["split_final"], f"raw_{linha['split']}/{linha['relativo']}"):
                 (linha["sha256"], linha["label_sha256"]) for _, linha in final.iterrows()}
    for _, linha in verificada["manifesto"].loc[verificada["manifesto"]["split"].ne("test")].iterrows():
        if esperados.get((linha["split"], linha["relativo"])) != (linha["sha256"], linha["label_sha256"]):
            raise RuntimeError("Arquivo de desenvolvimento mudou durante a cópia.")
    # dHash é um agrupador conservador, não prova identidade visual.
    for coluna in ["sha256", "dhash"]:
        cruzadas = verificada["manifesto"].groupby(coluna)["split"].nunique()
        if (cruzadas > 1).any():
            raise RuntimeError(f"Há grupos {coluna} atravessando conjuntos; preparação bloqueada.")
    relatorios.mkdir(parents=True, exist_ok=True)
    exportacao = final.copy()
    exportacao["ids_classes"] = exportacao["ids_classes"].map(lambda v: json.dumps(sorted(v)))
    exportacao.to_csv(relatorios / "grupos_e_splits.csv", index=False, encoding="utf-8-sig")
    exclusoes.to_csv(relatorios / "exclusoes_automaticas.csv", index=False, encoding="utf-8-sig")
    resumo = {"dataset": str(saida), "relatorios": str(relatorios), "semente": semente,
              "proporcao_val_desejada": proporcao, "test_sha256": referencia["sha256"],
              "imagens_por_split": verificada["resumo"].set_index("split")["imagens"].astype(int).to_dict(),
              "politica": "SHA+anotação iguais: uma cópia; dHash iguais: mesmo grupo; grupos relacionados ao TEST: fora do desenvolvimento"}
    salvar_json(relatorios / "resumo_preparacao.json", resumo)
    salvar_json(saida / "preparacao_concluida.json", resumo)
    return saida, resumo


def exportar_zip_validado(dataset, pasta_exportacao, classes=None, esperado_test=403, test_sha256_esperado=None):
    """Empacota somente uma base estruturalmente válida; nenhuma edição manual de CSV."""
    dataset, pasta_exportacao = Path(dataset), Path(pasta_exportacao)
    classes = classes or CLASSES_HOMEOBJECTS
    if ler_classes(dataset) != classes:
        raise RuntimeError("Mapa de classes incompatível para exportação.")
    pasta = pasta_exportacao / identificador_execucao()
    resultados = auditar_dataset(dataset, pasta / "auditoria_exportacao", classes, gerar_figuras=False)
    validar_resultados_auditoria(resultados)
    referencia = resumo_test(resultados["manifesto"], classes, esperado_test)
    if test_sha256_esperado is not None and referencia["sha256"] != test_sha256_esperado:
        raise RuntimeError("O TEST mudou desde a conferência. Exportação bloqueada.")
    arquivos = []
    for _, linha in resultados["manifesto"].iterrows():
        relativo = Path(linha["relativo"])
        arquivos.extend([
            {"origem": linha["imagem"], "arquivo": f"homeobjects_yolo/images/{linha['split']}/{relativo.as_posix()}", "sha256": linha["sha256"]},
            {"origem": linha["label"], "arquivo": f"homeobjects_yolo/labels/{linha['split']}/{relativo.with_suffix('.txt').as_posix()}", "sha256": linha["label_sha256"]},
        ])
    from .auditoria_dados import _calcular_sha256
    arquivos.append({"origem": str(dataset / "data.yaml"), "arquivo": "homeobjects_yolo/data.yaml",
                     "sha256": _calcular_sha256(dataset / "data.yaml")})
    metadados = {"versao": 1, "dataset_dir": "homeobjects_yolo", "classes": classes,
                 "imagens_por_split": resultados["resumo"].set_index("split")["imagens"].astype(int).to_dict(),
                 "test": referencia, "arquivos": [{k: v for k, v in a.items() if k != "origem"} for a in arquivos]}
    pacote = pasta / "homeobjects_yolo_validado.zip"
    with zipfile.ZipFile(pacote, "x", compression=zipfile.ZIP_DEFLATED) as arquivo_zip:
        for a in arquivos:
            if _calcular_sha256(Path(a["origem"])) != a["sha256"]:
                raise RuntimeError("Arquivo mudou durante a exportação.")
            arquivo_zip.write(a["origem"], a["arquivo"])
        arquivo_zip.writestr("verificacao_preparacao.json", json.dumps(metadados, ensure_ascii=False, indent=2))
    salvar_json(pasta / "verificacao_preparacao.json", metadados)
    return pacote
