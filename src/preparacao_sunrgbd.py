"""Integra o treino oficial SUN RGB-D ao HomeObjects, sem alterar o TEST.

Somente RGB e caixas 2D oficiais. Não cria rótulos por previsões do modelo.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from scipy.io import loadmat

from .auditoria_dados import (
    CLASSES_HOMEOBJECTS, _calcular_dhash, _calcular_sha256,
    auditar_dataset, validar_resultados_auditoria,
)
from .preparacao_dados import (
    _Uniao, exportar_zip_validado, identificador_execucao,
    proteger_dataset_existente, resumo_test, salvar_json,
)

URL_META = "https://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat"
URL_SPLIT_ZIP = "https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip"
URL_IMAGENS = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"

# IDs iguais aos do detector antigo. Categorias diferentes não são fundidas.
# plant/picture/painting não demonstram, por si só, vaso ou moldura.
ALIASES = {
    "bed": "bed", "bunk bed": "bed",
    "sofa": "sofa", "couch": "sofa",
    "chair": "chair", "office chair": "chair", "dining chair": "chair",
    "table": "table", "coffee table": "table", "endtable": "table",
    "end table": "table", "dining table": "table", "side table": "table",
    "lamp": "lamp", "table lamp": "lamp", "floor lamp": "lamp",
    "desk lamp": "lamp", "tv": "tv", "television": "tv",
    "laptop": "laptop", "notebook computer": "laptop",
    "wardrobe": "wardrobe", "armoire": "wardrobe",
    "window": "window", "door": "door",
    "potted plant": "potted plant",
    "photo frame": "photo frame", "picture frame": "photo frame",
    "photoframe": "photo frame", "pictureframe": "photo frame",
}
# Uma imagem com categoria potencialmente equivalente, mas ambígua, fica fora
# do experimento: não queremos ensinar que um alvo sem rótulo é fundo.
AMBIGUAS = {"plant", "plants", "artificial plant", "picture", "painting",
            "sofa chair", "desk", "computer"}


def normalizar_classe(valor):
    return re.sub(r"\s+", " ", str(valor).strip().lower().replace("_", " "))


def caminho_sun(valor):
    """Remove a raiz Unix original, preservando o caminho completo da cena."""
    partes = PurePosixPath(str(valor).replace("\\", "/")).parts
    if "SUNRGBD" not in partes or ".." in partes:
        raise ValueError(f"Caminho SUN RGB-D inválido: {valor}")
    indice = partes.index("SUNRGBD")
    return PurePosixPath(*partes[indice:]).as_posix().rstrip("/")


def objetos_meta(valor):
    if isinstance(valor, dict):
        return [valor]
    if isinstance(valor, (list, tuple, np.ndarray)):
        return list(valor)
    raise ValueError("Estrutura groundtruth2DBB não reconhecida.")


def ler_metadados(arquivo, esperado=10335):
    dados = loadmat(arquivo, simplify_cells=True)
    meta = dados.get("SUNRGBDMeta2DBB")
    if meta is None:
        raise ValueError("Arquivo sem SUNRGBDMeta2DBB.")
    registros = [meta] if isinstance(meta, dict) else list(meta)
    if esperado is not None and len(registros) != esperado:
        raise ValueError(f"Esperados {esperado} registros; encontrados {len(registros)}.")
    cenas = [caminho_sun(r["sequenceName"]) for r in registros]
    if len(set(cenas)) != len(cenas):
        raise ValueError("Metadados com caminhos de cena repetidos.")
    return registros


def ler_divisao_oficial(arquivo, registros):
    dados = loadmat(arquivo, simplify_cells=True)
    divisao = {}
    for chave, split in [("alltrain", "train"), ("alltest", "test")]:
        if chave not in dados:
            raise ValueError(f"allsplit.mat não contém {chave}.")
        valores = dados[chave]
        valores = [valores] if isinstance(valores, str) else list(valores)
        for valor in valores:
            cena = caminho_sun(valor)
            if cena in divisao:
                raise ValueError(f"Cena repetida no split oficial: {cena}")
            divisao[cena] = split
    cenas = {caminho_sun(r["sequenceName"]) for r in registros}
    if set(divisao) != cenas:
        raise ValueError("Divisão oficial e anotações não cobrem as mesmas cenas.")
    return divisao


def inventariar(registros, divisao):
    contagem = Counter()
    for r in registros:
        split = divisao[caminho_sun(r["sequenceName"])]
        for obj in objetos_meta(r["groundtruth2DBB"]):
            classe = normalizar_classe(obj["classname"])
            contagem[(split, classe)] += 1
    linhas = [{"split_oficial": s, "classe_sun": c, "objetos": n,
               "classe_projeto": ALIASES.get(c, ""),
               "tratamento": ("mapeada" if c in ALIASES else
                              "ambigua_excluir_imagem" if c in AMBIGUAS else
                              "outra_categoria")}
              for (s, c), n in sorted(contagem.items())]
    return pd.DataFrame(linhas)


def baixar_retornavel(url, destino):
    """Retoma .part quando há suporte a Range; verifica o comprimento final."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.is_file():
        return destino
    parcial = destino.with_name(destino.name + ".part")
    inicio = parcial.stat().st_size if parcial.exists() else 0
    headers = {"User-Agent": "TCC-SUNRGBD/1.0", "Accept-Encoding": "identity"}
    if inicio:
        headers["Range"] = f"bytes={inicio}-"
    try:
        resposta = urlopen(Request(url, headers=headers), timeout=120)
    except HTTPError as exc:
        if exc.code != 416 or not inicio:
            raise
        # Não promove um parcial por receber 416: compara com o tamanho remoto.
        with urlopen(Request(url, method="HEAD"), timeout=60) as r:
            total = int(r.headers.get("Content-Length", -1))
        if inicio != total:
            raise RuntimeError("Download parcial incompatível com o servidor.") from exc
        parcial.replace(destino)
        return destino
    with resposta:
        if resposta.status == 206:
            m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", resposta.headers.get("Content-Range", ""))
            if not m or int(m[1]) != inicio:
                raise RuntimeError("Servidor retornou uma faixa de download incorreta.")
            total = int(m[3])
            modo = "ab"
        elif resposta.status == 200:
            inicio, modo = 0, "wb"
            total = int(resposta.headers.get("Content-Length", -1))
        else:
            raise RuntimeError(f"Resposta HTTP inesperada: {resposta.status}")
        esperado = total - inicio
        if total <= 0:
            raise RuntimeError("Servidor não informou o tamanho: download não verificável.")
        livre = shutil.disk_usage(destino.parent).free
        if livre < esperado + 512 * 1024**2:
            raise OSError(f"Espaço insuficiente: precisa de pelo menos {esperado / 1024**3:.2f} GiB livres, mais margem.")
        print(f"Baixando {destino.name}: {inicio / 1024**2:.0f}/{total / 1024**2:.0f} MiB")
        escritos = inicio
        aviso = escritos + 128 * 1024**2
        with parcial.open(modo) as f:
            while bloco := resposta.read(1024 * 1024):
                f.write(bloco)
                escritos += len(bloco)
                if escritos >= aviso:
                    print(f"  {100 * escritos / total:.1f}% ({escritos / 1024**2:.0f} MiB)")
                    aviso = escritos + 128 * 1024**2
    if parcial.stat().st_size != total:
        raise RuntimeError("Download incompleto. Execute novamente para retomar.")
    parcial.replace(destino)
    return destino


class _ZipRemoto(io.RawIOBase):
    """Permite zipfile ler só um membro pequeno do toolbox oficial por HTTP."""
    def __init__(self, url):
        super().__init__()
        self.url, self.pos = url, 0
        with urlopen(Request(url, method="HEAD"), timeout=60) as r:
            self.tamanho = int(r.headers.get("Content-Length", 0))
        if self.tamanho <= 0:
            raise OSError("Tamanho do ZIP remoto indisponível.")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.tamanho if whence == 2 else 0)
        if self.pos < 0:
            raise OSError("Posição negativa no ZIP.")
        return self.pos

    def read(self, size=-1):
        size = min(self.tamanho - self.pos, self.tamanho if size < 0 else size)
        if size <= 0:
            return b""
        inicio, fim = self.pos, self.pos + size - 1
        req = Request(self.url, headers={"Range": f"bytes={inicio}-{fim}",
                                        "Accept-Encoding": "identity"})
        with urlopen(req, timeout=60) as r:
            esperado = f"bytes {inicio}-{fim}/{self.tamanho}"
            if r.status != 206 or r.headers.get("Content-Range") != esperado:
                raise OSError("Servidor não oferece a leitura parcial necessária.")
            bloco = r.read(size + 1)
        if len(bloco) != size:
            raise OSError("Leitura parcial incompleta.")
        self.pos += size
        return bloco


def obter_metadados(pasta_raw):
    pasta = Path(pasta_raw)
    pasta.mkdir(parents=True, exist_ok=True)
    meta = baixar_retornavel(URL_META, pasta / "SUNRGBDMeta2DBB_v2.mat")
    split = pasta / "allsplit.mat"
    if not split.exists():
        try:
            with _ZipRemoto(URL_SPLIT_ZIP) as remoto, zipfile.ZipFile(remoto) as z:
                membros = [n for n in z.namelist() if n.endswith("/allsplit.mat") and not n.startswith("__MACOSX/")]
                if len(membros) != 1:
                    raise ValueError("allsplit.mat não foi localizado de forma única.")
                raw = z.read(membros[0])  # zipfile também verifica o CRC do membro.
        except (OSError, HTTPError, zipfile.BadZipFile) as exc:
            print(f"Leitura parcial indisponível ({type(exc).__name__}); baixando o toolbox completo.")
            arquivo = baixar_retornavel(URL_SPLIT_ZIP, pasta / "SUNRGBDtoolbox.zip")
            with zipfile.ZipFile(arquivo) as z:
                membros = [n for n in z.namelist() if n.endswith("/allsplit.mat") and not n.startswith("__MACOSX/")]
                if len(membros) != 1:
                    raise ValueError("allsplit.mat não encontrado no toolbox.")
                raw = z.read(membros[0])
        parcial = split.with_suffix(".mat.part")
        parcial.write_bytes(raw)
        # Confere o MAT antes de colocá-lo no caminho definitivo.
        loadmat(parcial, simplify_cells=True)
        parcial.replace(split)
    registros = ler_metadados(meta)
    divisao = ler_divisao_oficial(split, registros)
    print("Divisão oficial conferida:", dict(Counter(divisao.values())))
    return registros, divisao


def candidatos_treino(registros, divisao):
    candidatos, exclusoes = [], []
    for indice, r in enumerate(registros, 1):
        cena = caminho_sun(r["sequenceName"])
        if divisao[cena] != "train":
            continue
        objetos = objetos_meta(r["groundtruth2DBB"])
        classes = {normalizar_classe(o["classname"]) for o in objetos}
        ambiguas = sorted(classes & AMBIGUAS)
        suspeitas = sorted(c for c in classes if "/" in c or "\\" in c)
        motivo = "classe_ambigua" if ambiguas else "classe_invalida" if suspeitas else "sem_alvo_mapeado" if not classes.intersection(ALIASES) else ""
        if motivo:
            exclusoes.append({"sun_id": indice, "cena": cena, "motivo": motivo,
                              "detalhe": "; ".join(ambiguas or suspeitas)})
            continue
        rgb = caminho_sun(r["rgbpath"])
        esperado = PurePosixPath(cena, "image", str(r["rgbname"])).as_posix()
        if rgb != esperado:
            raise ValueError(f"rgbpath não corresponde à cena {indice}.")
        candidatos.append({"sun_id": indice, "cena": cena, "rgb_relativo": rgb,
                           "objetos": objetos})
    return candidatos, exclusoes


def obter_imagens(pasta_raw, candidatos):
    """Reutiliza a estrutura oficial; extrai só RGB necessário do ZIP completo."""
    pasta = Path(pasta_raw)
    ausentes = [c for c in candidatos if not (pasta / c["rgb_relativo"]).is_file()]
    if not ausentes:
        print("Todas as imagens candidatas já estão disponíveis.")
        return
    arquivo = baixar_retornavel(URL_IMAGENS, pasta / "SUNRGBD.zip")
    with zipfile.ZipFile(arquivo) as z:
        nomes = {i.filename: i for i in z.infolist()}
        nao_localizados = [c["rgb_relativo"] for c in ausentes if c["rgb_relativo"] not in nomes]
        if nao_localizados:
            raise FileNotFoundError(f"ZIP oficial não contém {len(nao_localizados)} RGBs esperados. Exemplo: {nao_localizados[0]}")
        necessario = sum(nomes[c["rgb_relativo"]].file_size for c in ausentes)
        if shutil.disk_usage(pasta).free < necessario + 512 * 1024**2:
            raise OSError("Espaço insuficiente para extrair as imagens RGB.")
        for numero, c in enumerate(ausentes, 1):
            destino = (pasta / c["rgb_relativo"]).resolve()
            if pasta.resolve() not in destino.parents:
                raise ValueError("Caminho de imagem fora da pasta raw.")
            destino.parent.mkdir(parents=True, exist_ok=True)
            parcial = destino.with_suffix(destino.suffix + ".part")
            with z.open(c["rgb_relativo"]) as origem, parcial.open("wb") as saida:
                shutil.copyfileobj(origem, saida)
            parcial.replace(destino)
            if numero % 500 == 0:
                print(f"RGBs extraídos: {numero}/{len(ausentes)}")
    print("Imagens RGB prontas; profundidade e fullres não foram extraídos.")


def converter_caixa(xywh, largura, altura):
    """gtBb2D: origem MATLAB em 1; converte para bordas contínuas em 0.

    x0=x-1, y0=y-1; a largura/altura permanecem as anotadas. Recorta só
    nas bordas reais. Não redimensiona a imagem nem assume 640x480.
    """
    valores = np.asarray(xywh, dtype=float).reshape(-1)
    if len(valores) != 4 or not np.isfinite(valores).all():
        raise ValueError("Caixa 2D com valores inválidos.")
    x, y, w, h = valores
    if largura <= 0 or altura <= 0 or w <= 0 or h <= 0:
        raise ValueError("Dimensões ou caixa 2D sem área positiva.")
    original = [x - 1, y - 1, x - 1 + w, y - 1 + h]
    caixa = [max(0.0, original[0]), max(0.0, original[1]),
             min(float(largura), original[2]), min(float(altura), original[3])]
    x1, y1, x2, y2 = caixa
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Caixa anotada inteiramente fora da imagem real.")
    return [(x1 + x2) / (2 * largura), (y1 + y2) / (2 * altura),
            (x2 - x1) / largura, (y2 - y1) / altura], caixa != original


def grupo_captura(cena):
    """Agrupa cenas SUN3D e sessões Kinect2 identificáveis nos caminhos.

    NYU/RealSense sem ID de sala confiável: sequência + hashes. Não se
    afirma que estes grupos recuperam todas as salas físicas da base.
    """
    partes = PurePosixPath(cena).parts
    if "sun3ddata" in partes:
        i = partes.index("sun3ddata")
        return "/".join(partes[:i + 2])
    nome = partes[-1]
    m = re.fullmatch(r"\d+_(\d{4}-\d{2}-\d{2})_(\d{2})-\d{2}-\d{2}_(\d+)_rgbf.*", nome)
    if m:
        return f"kv2/{m[1]}/{m[2]}/{m[3]}"
    return cena


def converter_candidatos(candidatos, pasta_raw, pasta_relatorios):
    pasta_raw, relatorios = Path(pasta_raw), Path(pasta_relatorios)
    relatorios.mkdir(parents=True, exist_ok=True)
    linhas, exclusoes, caixas = [], [], []
    ids = {nome: i for i, nome in CLASSES_HOMEOBJECTS.items()}
    labels = relatorios / "labels_sun_intermediarios"
    labels.mkdir(exist_ok=True)
    for candidato in candidatos:
        caminho = pasta_raw / candidato["rgb_relativo"]
        if not caminho.is_file():
            raise FileNotFoundError(f"Imagem RGB ausente: {caminho}")
        try:
            with Image.open(caminho) as im:
                largura, altura = im.size
                im.verify()
            with Image.open(caminho) as im:
                im.load()
                dhash = _calcular_dhash(im)
        except (OSError, ValueError) as exc:
            exclusoes.append({"sun_id": candidato["sun_id"], "cena": candidato["cena"],
                              "motivo": "imagem_corrompida", "detalhe": str(exc)})
            continue
        anotacoes, classes, invalidas = [], [], []
        for obj in candidato["objetos"]:
            origem = normalizar_classe(obj["classname"])
            if origem not in ALIASES:
                continue
            classe = ids[ALIASES[origem]]
            try:
                bbox, recortada = converter_caixa(obj["gtBb2D"], largura, altura)
            except ValueError as exc:
                invalidas.append(str(exc))
                continue
            anotacoes.append((classe, *bbox))
            classes.append(classe)
            caixas.append({"sun_id": candidato["sun_id"], "classe_sun": origem,
                           "classe_projeto": CLASSES_HOMEOBJECTS[classe],
                           "largura_imagem": largura, "altura_imagem": altura,
                           "recortada": recortada, "bbox_yolo": json.dumps(bbox)})
        # Não remove um alvo inválido e mantém a imagem como se fosse fundo.
        if invalidas:
            exclusoes.append({"sun_id": candidato["sun_id"], "cena": candidato["cena"],
                              "motivo": "bbox_alvo_invalida_excluir_imagem",
                              "detalhe": "; ".join(invalidas)})
            continue
        if not anotacoes:
            raise AssertionError("Candidato positivo não produziu anotações.")
        anotacoes = sorted(set(anotacoes))
        label = labels / f"sun_{candidato['sun_id']:06d}.txt"
        label.write_text("\n".join(f"{a[0]} " + " ".join(f"{v:.10f}" for v in a[1:])
                                    for a in anotacoes) + "\n", encoding="utf-8")
        linhas.append({"sun_id": candidato["sun_id"], "cena": candidato["cena"],
                       "imagem": str(caminho), "label": str(label),
                       "sha256": _calcular_sha256(caminho), "dhash": dhash,
                       "label_sha256": _calcular_sha256(label),
                       "largura": largura, "altura": altura,
                       "ids_classes": [a[0] for a in anotacoes],
                       "captura": grupo_captura(candidato["cena"])})
    if not linhas:
        raise RuntimeError("Não restaram imagens SUN estruturalmente válidas.")
    pd.DataFrame(caixas).to_csv(relatorios / "conversao_caixas.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(linhas), exclusoes


def agrupar_e_separar(sun, home, proporcao=0.20, semente=42):
    """Mantém Home train/val; SUN relacionada ao Home é excluída por grupo."""
    if not 0 < proporcao < 1:
        raise ValueError("A proporção de validação deve estar entre 0 e 1.")
    sun = sun.sort_values("sun_id").reset_index(drop=True).copy()
    uf = _Uniao(len(sun))
    for coluna in ["captura", "sha256", "dhash"]:
        primeiro = {}
        for i, valor in enumerate(sun[coluna]):
            if valor in primeiro:
                uf.juntar(i, primeiro[valor])
            else:
                primeiro[valor] = i
    sun["grupo"] = [uf.raiz(i) for i in range(len(sun))]
    tocando_home = sun["sha256"].isin(set(home["sha256"])) | sun["dhash"].isin(set(home["dhash"]))
    excluidos = set(sun.loc[tocando_home, "grupo"])
    exclusoes = [{"sun_id": r["sun_id"], "cena": r["cena"],
                  "motivo": "grupo_relacionado_ao_home", "detalhe": str(r["grupo"])}
                 for _, r in sun.loc[sun["grupo"].isin(excluidos)].iterrows()]
    sun = sun.loc[~sun["grupo"].isin(excluidos)].copy()
    grupos = {g: {"n": len(r), "classes": Counter(c for ids in r["ids_classes"] for c in ids)}
              for g, r in sun.groupby("grupo")}
    if len(grupos) < 2:
        raise RuntimeError("Menos de dois grupos SUN independentes disponíveis para train/val.")
    totais = Counter(c for ids in sun["ids_classes"] for c in ids)
    alvo = max(1, round(len(sun) * proporcao))
    melhor = None
    for tentativa in range(128):
        ordem = sorted(grupos)
        random.Random(semente + tentativa).shuffle(ordem)
        escolhidos, n_val = set(), 0
        for g in ordem:
            if n_val < alvo:
                escolhidos.add(g)
                n_val += grupos[g]["n"]
        if not escolhidos or len(escolhidos) == len(grupos):
            continue
        classes_val = Counter()
        for g in escolhidos:
            classes_val.update(grupos[g]["classes"])
        custo = abs(n_val / len(sun) - proporcao)
        custo += sum(abs(classes_val[c] / n - proporcao) for c, n in totais.items()) / max(len(totais), 1)
        if melhor is None or custo < melhor[0]:
            melhor = custo, escolhidos
    if melhor is None:
        raise RuntimeError("Não foi possível separar SUN por grupos.")
    sun["split_final"] = sun["grupo"].map(lambda g: "val" if g in melhor[1] else "train")
    if sun.groupby("grupo")["split_final"].nunique().max() != 1:
        raise AssertionError("Um grupo SUN atravessou train e val.")
    return sun, exclusoes


def criar_base_mista(dataset_home, pasta_raw, raiz_projeto, registros, divisao,
                     proporcao_sun_val=0.20, semente=42, esperado_test=403):
    raiz, home_dir = Path(raiz_projeto), Path(dataset_home)
    identificador = identificador_execucao()
    relatorios = raiz / "reports/preparacao_sunrgbd" / identificador
    relatorios.mkdir(parents=True, exist_ok=False)
    home, referencia = proteger_dataset_existente(home_dir, relatorios, esperado=esperado_test)
    manifesto_home = home["manifesto"]
    for coluna in ["sha256", "dhash"]:
        if (manifesto_home.groupby(coluna)["split"].nunique() > 1).any():
            raise RuntimeError("Home contém grupos atravessando conjuntos. Use a base validada pelo notebook 02.")
    inventario = inventariar(registros, divisao)
    inventario.to_csv(relatorios / "inventario_classes_sun.csv", index=False, encoding="utf-8-sig")
    candidatos, excluidos = candidatos_treino(registros, divisao)
    sun, excluidos_caixas = converter_candidatos(candidatos, pasta_raw, relatorios)
    sun, excluidos_grupos = agrupar_e_separar(sun, manifesto_home, proporcao_sun_val, semente)
    pd.DataFrame(excluidos + excluidos_caixas + excluidos_grupos).to_csv(
        relatorios / "exclusoes_automaticas.csv", index=False, encoding="utf-8-sig")
    saida = raiz / "data/processed/homeobjects_sunrgbd" / identificador
    saida.mkdir(parents=True, exist_ok=False)
    destinos_esperados = {}
    for _, r in manifesto_home.iterrows():
        # TEST conserva todos os caminhos e bytes. Home train/val também.
        relativo = Path(r["relativo"])
        for tipo, fonte, destino_rel in [("images", r["imagem"], relativo),
                                        ("labels", r["label"], relativo.with_suffix(".txt"))]:
            destino = saida / tipo / r["split"] / destino_rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fonte, destino)
        destinos_esperados[(r["split"], relativo.as_posix())] = (r["sha256"], r["label_sha256"])
    for _, r in sun.iterrows():
        relativo = Path("sunrgbd") / f"sun_{r['sun_id']:06d}{Path(r['imagem']).suffix.lower()}"
        for tipo, fonte, destino_rel in [("images", r["imagem"], relativo),
                                        ("labels", r["label"], relativo.with_suffix(".txt"))]:
            destino = saida / tipo / r["split_final"] / destino_rel
            if destino.exists():
                raise RuntimeError(f"Colisão de nome no dataset misto: {destino}")
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fonte, destino)
        destinos_esperados[(r["split_final"], relativo.as_posix())] = (r["sha256"], r["label_sha256"])
    nomes = CLASSES_HOMEOBJECTS
    configuracao = {"path": saida.as_posix(), "train": "images/train", "val": "images/val",
                    "test": "images/test", "names": nomes}
    (saida / "data.yaml").write_text(yaml.safe_dump(configuracao, sort_keys=False, allow_unicode=True), encoding="utf-8")
    validada = auditar_dataset(saida, relatorios / "auditoria_final", nomes, gerar_figuras=False)
    validar_resultados_auditoria(validada)
    manifesto = validada["manifesto"]
    if resumo_test(manifesto, nomes, esperado_test)["sha256"] != referencia["sha256"]:
        raise RuntimeError("A cópia do TEST mudou. Preparação bloqueada.")
    for _, r in manifesto.iterrows():
        if destinos_esperados.get((r["split"], r["relativo"])) != (r["sha256"], r["label_sha256"]):
            raise RuntimeError("Imagem ou anotação mudou durante a cópia.")
    for coluna in ["sha256", "dhash"]:
        if (manifesto.groupby(coluna)["split"].nunique() > 1).any():
            raise RuntimeError(f"Grupo {coluna} atravessou conjuntos.")
    origens = []
    for _, r in manifesto.iterrows():
        origem = "sunrgbd" if r["relativo"].startswith("sunrgbd/") else "homeobjects"
        origens.append({"split": r["split"], "relativo": r["relativo"], "origem": origem,
                        "sha256": r["sha256"], "label_sha256": r["label_sha256"]})
    pd.DataFrame(origens).to_csv(relatorios / "origem_das_imagens.csv", index=False, encoding="utf-8-sig")
    sun.assign(ids_classes=sun["ids_classes"].map(json.dumps)).to_csv(
        relatorios / "grupos_sun_e_splits.csv", index=False, encoding="utf-8-sig")
    distribuicao = []
    for split in ["train", "val"]:
        for origem in ["homeobjects", "sunrgbd"]:
            subset = manifesto.loc[manifesto["split"].eq(split)
                                   & manifesto["relativo"].str.startswith("sunrgbd/").eq(origem == "sunrgbd")]
            # O manifesto tem apenas presença de classes; a tabela de caixas
            # conserva cada instância e fornece a distribuição de objetos.
            caixas_origem = validada["caixas"].loc[
                validada["caixas"]["split"].eq(split)
                & validada["caixas"]["imagem"].isin(set(subset["imagem"]))]
            contagem = Counter(caixas_origem["id_classe"])
            distribuicao += [{"split": split, "origem": origem, "id_classe": c, "classe": n,
                              "objetos": contagem[c]} for c, n in nomes.items()]
    pd.DataFrame(distribuicao).to_csv(relatorios / "distribuicao_classes_mista.csv", index=False, encoding="utf-8-sig")
    resumo = {"dataset": str(saida), "relatorios": str(relatorios), "home_origem": str(home_dir),
              "semente": semente, "proporcao_sun_val_desejada": proporcao_sun_val,
              "test_sha256": referencia["sha256"], "test_imagens": esperado_test,
              "imagens_por_split": manifesto.groupby("split").size().to_dict(),
              "imagens_por_origem": pd.DataFrame(origens).groupby(["split", "origem"]).size().rename("imagens").reset_index().to_dict("records"),
              "sun_split_oficial": dict(Counter(divisao.values())),
              "sun_test_oficial_utilizado": False,
              "metadados_sha256": _calcular_sha256(Path(pasta_raw) / "SUNRGBDMeta2DBB_v2.mat"),
              "split_oficial_sha256": _calcular_sha256(Path(pasta_raw) / "allsplit.mat"),
              "aliases": ALIASES, "classes_ambiguas_excluem_imagem": sorted(AMBIGUAS),
              "convencao_bbox": "xywh MATLAB: x0=x-1, y0=y-1; recorte nas dimensões reais",
              "agrupamento": "cena SUN3D/sessão Kinect2 quando identificável; SHA e dHash iguais; grupos relacionados ao Home excluídos",
              "limite_agrupamento": "hashes não detectam toda semelhança; caminhos não identificam todas as salas físicas",
              "exclusoes_sun": len(excluidos) + len(excluidos_caixas) + len(excluidos_grupos)}
    salvar_json(relatorios / "resumo_preparacao_sunrgbd.json", resumo)
    salvar_json(saida / "preparacao_concluida.json", resumo)
    # Reutiliza o formato de pacote já verificado no projeto.
    pacote_original = exportar_zip_validado(saida, raiz / "exports/datasets_sunrgbd",
                                            esperado_test=esperado_test,
                                            test_sha256_esperado=referencia["sha256"])
    pacote = pacote_original.with_name("homeobjects_sunrgbd_validado.zip")
    resumo["zip_colab"] = str(pacote)
    salvar_json(relatorios / "resumo_preparacao_sunrgbd.json", resumo)
    extras = {
        "resumo_preparacao_sunrgbd.json": relatorios / "resumo_preparacao_sunrgbd.json",
        "origem_das_imagens.csv": relatorios / "origem_das_imagens.csv",
        "codigo/src/validacao_alertas.py": Path(__file__).with_name("validacao_alertas.py"),
        "codigo/src/politica_alertas.py": Path(__file__).with_name("politica_alertas.py"),
    }
    # Recria o ZIP com uma única cópia do manifesto, incluindo hashes dos
    # relatórios e da mesma política utilizada na validação do Colab.
    with zipfile.ZipFile(pacote_original) as original, zipfile.ZipFile(
            pacote, "x", compression=zipfile.ZIP_DEFLATED) as z:
        metadados = json.loads(original.read("verificacao_preparacao.json"))
        for item in original.infolist():
            if item.filename == "verificacao_preparacao.json":
                continue
            with original.open(item) as fonte, z.open(item.filename, "w", force_zip64=True) as destino:
                shutil.copyfileobj(fonte, destino)
        for nome, arquivo in extras.items():
            z.write(arquivo, nome)
            metadados["arquivos"].append({"arquivo": nome, "sha256": _calcular_sha256(arquivo)})
        z.writestr("verificacao_preparacao.json", json.dumps(metadados, ensure_ascii=False, indent=2))
    pacote_original.unlink()  # Só o ZIP intermediário recém-criado nesta execução.
    print("Base mista validada:", saida)
    print("Pacote para o Colab:", pacote)
    return saida, pacote, resumo


def gerar_previa(sun, destino, quantidade=8, semente=42):
    """Amostra automática de conferência, sem editar rótulos ou escolher TEST."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(sun)))
    random.Random(semente).shuffle(indices)
    salvos = []
    for i in indices[:quantidade]:
        r = sun.iloc[i]
        with Image.open(r["imagem"]) as original:
            imagem = original.convert("RGB")
        draw = ImageDraw.Draw(imagem)
        for linha in Path(r["label"]).read_text(encoding="utf-8").splitlines():
            c, xc, yc, w, h = map(float, linha.split())
            x1, y1 = (xc - w / 2) * imagem.width, (yc - h / 2) * imagem.height
            x2, y2 = (xc + w / 2) * imagem.width, (yc + h / 2) * imagem.height
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 2, max(0, y1)), CLASSES_HOMEOBJECTS[int(c)], fill="yellow")
        caminho = destino / f"sun_{r['sun_id']:06d}.jpg"
        imagem.save(caminho, quality=92)
        salvos.append(caminho)
    return salvos
