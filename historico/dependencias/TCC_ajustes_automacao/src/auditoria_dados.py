from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


URL_HOMEOBJECTS = (
    "https://github.com/ultralytics/assets/releases/download/"
    "v0.0.0/homeobjects-3K.zip"
)

CLASSES_HOMEOBJECTS = {
    0: "bed",
    1: "sofa",
    2: "chair",
    3: "table",
    4: "lamp",
    5: "tv",
    6: "laptop",
    7: "wardrobe",
    8: "window",
    9: "door",
    10: "potted plant",
    11: "photo frame",
}

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLUNAS_PROBLEMAS = ["split", "imagem", "label", "tipo", "detalhe", "severidade"]
COLUNAS_DUPLICIDADES = ["tipo_hash", "hash", "quantidade", "splits", "entre_splits", "arquivos"]
COLUNAS_MANIFESTO = ["split", "relativo", "imagem", "label", "sha256", "dhash",
                    "label_sha256", "largura", "altura", "ids_classes", "caixas_validas"]


def validar_resultados_auditoria(resultados: dict[str, pd.DataFrame]) -> None:
    """Bloqueia erros estruturais; labels vazios não são erros por si só."""
    problemas = resultados["problemas"]
    criticos = problemas.loc[problemas["severidade"].eq("critica")]
    if not criticos.empty:
        resumo = criticos["tipo"].value_counts().to_dict()
        raise RuntimeError(
            f"Preparação bloqueada: {len(criticos)} problema(s) crítico(s): {resumo}. "
            "Consulte problemas_anotacoes.csv. Não crie nem complete anotações por palpite."
        )


def localizar_raiz_dataset(pasta_busca: Path) -> Path | None:
    """Localiza a pasta que contém os diretórios images e labels."""
    pasta_busca = Path(pasta_busca)
    candidatos = [pasta_busca]

    if pasta_busca.exists():
        candidatos.extend(p for p in pasta_busca.rglob("*") if p.is_dir())

    for candidato in candidatos:
        if (candidato / "images").is_dir() and (candidato / "labels").is_dir():
            return candidato

    return None


def _extrair_zip_seguro(arquivo_zip: Path, destino: Path) -> None:
    """Extrai o ZIP impedindo que arquivos sejam escritos fora do destino."""
    destino_resolvido = destino.resolve()

    with zipfile.ZipFile(arquivo_zip) as zip_ref:
        for membro in zip_ref.infolist():
            caminho_final = (destino / membro.filename).resolve()
            if destino_resolvido not in caminho_final.parents and caminho_final != destino_resolvido:
                raise ValueError(f"Caminho inseguro encontrado no ZIP: {membro.filename}")

        zip_ref.extractall(destino)


def baixar_homeobjects(
    pasta_raw: str | Path,
    url: str = URL_HOMEOBJECTS,
) -> Path:
    """Baixa e extrai o HomeObjects-3K, reutilizando arquivos existentes."""
    pasta_raw = Path(pasta_raw)
    pasta_raw.mkdir(parents=True, exist_ok=True)

    raiz_existente = localizar_raiz_dataset(pasta_raw)
    if raiz_existente is not None:
        print(f"Dataset já encontrado em: {raiz_existente}")
        return raiz_existente

    arquivo_zip = pasta_raw / "homeobjects-3K.zip"
    arquivo_parcial = pasta_raw / "homeobjects-3K.zip.part"

    if not arquivo_zip.exists():
        print("Baixando o HomeObjects-3K. O arquivo possui algumas centenas de MB...")

        requisicao = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(requisicao, timeout=300) as resposta:
            tamanho_total = int(resposta.headers.get("content-length", 0))
            baixados = 0
            proximo_aviso = 25 * 1024 * 1024

            with arquivo_parcial.open("wb") as arquivo_saida:
                while True:
                    bloco = resposta.read(1024 * 1024)
                    if not bloco:
                        break
                    arquivo_saida.write(bloco)
                    baixados += len(bloco)

                    if baixados >= proximo_aviso:
                        if tamanho_total:
                            percentual = 100 * baixados / tamanho_total
                            print(f"  Baixados: {baixados / 1024**2:.0f} MB ({percentual:.1f}%)")
                        else:
                            print(f"  Baixados: {baixados / 1024**2:.0f} MB")
                        proximo_aviso += 25 * 1024 * 1024

        arquivo_parcial.replace(arquivo_zip)
    else:
        print(f"ZIP já encontrado em: {arquivo_zip}")

    pasta_extracao = pasta_raw / "homeobjects_extraido"
    pasta_extracao.mkdir(parents=True, exist_ok=True)

    print("Extraindo o dataset...")
    _extrair_zip_seguro(arquivo_zip, pasta_extracao)

    raiz_dataset = localizar_raiz_dataset(pasta_extracao)
    if raiz_dataset is None:
        raise FileNotFoundError(
            "O ZIP foi extraído, mas não foi encontrada uma pasta com images/ e labels/."
        )

    print(f"Dataset pronto em: {raiz_dataset}")
    return raiz_dataset


def _listar_imagens(pasta: Path) -> list[Path]:
    return sorted(
        caminho
        for caminho in pasta.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM
    )


def _calcular_sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def _calcular_dhash(imagem: Image.Image) -> str:
    """Calcula um hash perceptual simples com diferenças de luminosidade."""
    reduzida = imagem.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(reduzida.getdata())
    bits = []

    for linha in range(8):
        inicio = linha * 9
        for coluna in range(8):
            bits.append(pixels[inicio + coluna] > pixels[inicio + coluna + 1])

    valor = sum(int(bit) << indice for indice, bit in enumerate(bits))
    return f"{valor:016x}"


def _registrar_duplicidades(
    grupos: dict[str, list[tuple[str, Path]]],
    tipo_hash: str,
) -> list[dict[str, Any]]:
    registros: list[dict[str, Any]] = []

    for valor_hash, itens in grupos.items():
        if len(itens) < 2:
            continue

        splits = sorted({split for split, _ in itens})
        registros.append(
            {
                "tipo_hash": tipo_hash,
                "hash": valor_hash,
                "quantidade": len(itens),
                "splits": ", ".join(splits),
                "entre_splits": len(splits) > 1,
                "arquivos": " | ".join(str(caminho) for _, caminho in itens),
            }
        )

    return registros


def _ler_anotacoes_yolo(
    caminho_label: Path,
    caminho_imagem: Path,
    split: str,
    largura_imagem: int,
    altura_imagem: int,
    classes: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    caixas: list[dict[str, Any]] = []
    problemas: list[dict[str, Any]] = []

    if not caminho_label.exists():
        problemas.append(
            {
                "split": split,
                "imagem": str(caminho_imagem),
                "label": str(caminho_label),
                "tipo": "sem_arquivo_anotacao",
                "detalhe": "Pode ser uma imagem negativa ou uma anotação ausente; revisar.",
            }
        )
        return caixas, problemas

    try:
        linhas = caminho_label.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as erro:
        problemas.append(
            {
                "split": split,
                "imagem": str(caminho_imagem),
                "label": str(caminho_label),
                "tipo": "erro_codificacao_label",
                "detalhe": str(erro),
            }
        )
        return caixas, problemas

    linhas_validas = [linha.strip() for linha in linhas if linha.strip()]
    if not linhas_validas:
        problemas.append(
            {
                "split": split,
                "imagem": str(caminho_imagem),
                "label": str(caminho_label),
                "tipo": "label_vazio",
                "detalhe": "Imagem sem objetos rotulados; pode ser uma amostra negativa válida.",
            }
        )

    for numero_linha, linha in enumerate(linhas_validas, start=1):
        partes = linha.split()
        if len(partes) != 5:
            problemas.append(
                {
                    "split": split,
                    "imagem": str(caminho_imagem),
                    "label": str(caminho_label),
                    "tipo": "quantidade_campos_invalida",
                    "detalhe": f"Linha {numero_linha}: esperado 5 campos, recebido {len(partes)}.",
                }
            )
            continue

        try:
            classe_numerica = float(partes[0])
            if not math.isfinite(classe_numerica) or not classe_numerica.is_integer():
                raise ValueError("O ID de classe precisa ser inteiro e finito.")
            id_classe = int(classe_numerica)
            centro_x, centro_y, largura, altura = map(float, partes[1:])
        except (ValueError, OverflowError):
            problemas.append(
                {
                    "split": split,
                    "imagem": str(caminho_imagem),
                    "label": str(caminho_label),
                    "tipo": "valor_nao_numerico",
                    "detalhe": f"Linha {numero_linha}: {linha}",
                }
            )
            continue

        if id_classe not in classes:
            problemas.append(
                {
                    "split": split,
                    "imagem": str(caminho_imagem),
                    "label": str(caminho_label),
                    "tipo": "classe_desconhecida",
                    "detalhe": f"Linha {numero_linha}: classe {id_classe}.",
                }
            )
            continue

        coordenadas = [centro_x, centro_y, largura, altura]
        if not all(0 <= valor <= 1 for valor in coordenadas) or largura <= 0 or altura <= 0:
            problemas.append(
                {
                    "split": split,
                    "imagem": str(caminho_imagem),
                    "label": str(caminho_label),
                    "tipo": "coordenada_normalizada_invalida",
                    "detalhe": f"Linha {numero_linha}: {linha}",
                }
            )
            continue

        x1 = centro_x - largura / 2
        y1 = centro_y - altura / 2
        x2 = centro_x + largura / 2
        y2 = centro_y + altura / 2
        tolerancia = 1e-5

        if min(x1, y1) < -tolerancia or max(x2, y2) > 1 + tolerancia:
            problemas.append(
                {
                    "split": split,
                    "imagem": str(caminho_imagem),
                    "label": str(caminho_label),
                    "tipo": "caixa_fora_da_imagem",
                    "detalhe": f"Linha {numero_linha}: {linha}",
                }
            )
            continue

        caixas.append(
            {
                "split": split,
                "imagem": str(caminho_imagem),
                "label": str(caminho_label),
                "id_classe": id_classe,
                "classe": classes[id_classe],
                "centro_x": centro_x,
                "centro_y": centro_y,
                "largura_relativa": largura,
                "altura_relativa": altura,
                "area_relativa": largura * altura,
                "proporcao_largura_altura": largura / altura,
                "largura_imagem": largura_imagem,
                "altura_imagem": altura_imagem,
            }
        )

    return caixas, problemas


def auditar_dataset(
    raiz_dataset: str | Path,
    pasta_relatorios: str | Path,
    classes: dict[int, str] | None = None,
    gerar_figuras: bool = True,
    bloquear_erros: bool = False,
) -> dict[str, pd.DataFrame]:
    """Audita imagens, labels YOLO, distribuição, caixas e duplicidades."""
    raiz_dataset = Path(raiz_dataset)
    pasta_relatorios = Path(pasta_relatorios)
    classes = classes or CLASSES_HOMEOBJECTS

    pasta_tabelas = pasta_relatorios / "tables"
    pasta_figuras = pasta_relatorios / "figures"
    pasta_tabelas.mkdir(parents=True, exist_ok=True)
    pasta_figuras.mkdir(parents=True, exist_ok=True)

    caixas: list[dict[str, Any]] = []
    problemas: list[dict[str, Any]] = []
    resumo_splits: list[dict[str, Any]] = []
    hashes_sha256: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    hashes_perceptuais: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    classes_por_imagem: dict[str, set[str]] = defaultdict(set)
    manifesto: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        pasta_imagens = raiz_dataset / "images" / split
        pasta_labels = raiz_dataset / "labels" / split

        if not pasta_imagens.exists():
            continue

        imagens = _listar_imagens(pasta_imagens)
        imagens_corrompidas = 0
        labels_ausentes = 0
        labels_vazios = 0

        print(f"Auditando {split}: {len(imagens)} imagens")
        for indice_imagem, caminho_imagem in enumerate(imagens, start=1):
            if indice_imagem == 1 or indice_imagem % 250 == 0 or indice_imagem == len(imagens):
                print(f"  {split}: {indice_imagem}/{len(imagens)}")
            relativo = caminho_imagem.relative_to(pasta_imagens)
            caminho_label = (pasta_labels / relativo).with_suffix(".txt")

            try:
                with Image.open(caminho_imagem) as imagem:
                    imagem.verify()
                with Image.open(caminho_imagem) as imagem:
                    largura_imagem, altura_imagem = imagem.size
                    hash_perceptual = _calcular_dhash(imagem)
            except Exception as erro:
                imagens_corrompidas += 1
                problemas.append(
                    {
                        "split": split,
                        "imagem": str(caminho_imagem),
                        "label": str(caminho_label),
                        "tipo": "imagem_corrompida",
                        "detalhe": str(erro),
                    }
                )
                manifesto.append({"split": split, "relativo": relativo.as_posix(),
                                  "imagem": str(caminho_imagem), "label": str(caminho_label),
                                  "sha256": _calcular_sha256(caminho_imagem), "dhash": None,
                                  "label_sha256": _calcular_sha256(caminho_label) if caminho_label.is_file() else None,
                                  "largura": None, "altura": None, "ids_classes": "[]", "caixas_validas": 0})
                continue

            hash_sha256 = _calcular_sha256(caminho_imagem)
            hashes_sha256[hash_sha256].append((split, caminho_imagem))
            hashes_perceptuais[hash_perceptual].append((split, caminho_imagem))

            caixas_imagem, problemas_imagem = _ler_anotacoes_yolo(
                caminho_label=caminho_label,
                caminho_imagem=caminho_imagem,
                split=split,
                largura_imagem=largura_imagem,
                altura_imagem=altura_imagem,
                classes=classes,
            )
            caixas.extend(caixas_imagem)
            problemas.extend(problemas_imagem)
            manifesto.append({
                "split": split, "relativo": relativo.as_posix(),
                "imagem": str(caminho_imagem), "label": str(caminho_label),
                "sha256": hash_sha256, "dhash": hash_perceptual,
                "label_sha256": _calcular_sha256(caminho_label) if caminho_label.is_file() else None,
                "largura": largura_imagem, "altura": altura_imagem,
                "ids_classes": json.dumps(sorted({c["id_classe"] for c in caixas_imagem})),
                "caixas_validas": len(caixas_imagem),
            })

            if not caminho_label.exists():
                labels_ausentes += 1
            elif any(p["tipo"] == "label_vazio" for p in problemas_imagem):
                labels_vazios += 1

            for caixa in caixas_imagem:
                classes_por_imagem[str(caminho_imagem)].add(caixa["classe"])

        resumo_splits.append(
            {
                "split": split,
                "imagens": len(imagens),
                "imagens_corrompidas": imagens_corrompidas,
                "labels_ausentes": labels_ausentes,
                "labels_vazios": labels_vazios,
            }
        )

    if not manifesto:
        problemas.append({"split": "-", "imagem": "", "label": "",
                          "tipo": "dataset_sem_imagens", "detalhe": "Nenhuma imagem encontrada."})
    for problema in problemas:
        problema["severidade"] = "aviso" if problema["tipo"] == "label_vazio" else "critica"
    df_caixas = pd.DataFrame(caixas, columns=[
        "split", "imagem", "label", "id_classe", "classe", "centro_x", "centro_y",
        "largura_relativa", "altura_relativa", "area_relativa", "proporcao_largura_altura",
        "largura_imagem", "altura_imagem",
    ])
    df_problemas = pd.DataFrame(problemas, columns=COLUNAS_PROBLEMAS)
    df_manifesto = pd.DataFrame(manifesto, columns=COLUNAS_MANIFESTO)
    df_resumo = pd.DataFrame(resumo_splits)

    contador_instancias = Counter(df_caixas.get("classe", pd.Series(dtype=str)))
    contador_imagens = Counter()
    for nomes_classes in classes_por_imagem.values():
        contador_imagens.update(nomes_classes)

    df_classes = pd.DataFrame(
        [
            {
                "id_classe": id_classe,
                "classe": nome_classe,
                "instancias": contador_instancias[nome_classe],
                "imagens_com_classe": contador_imagens[nome_classe],
            }
            for id_classe, nome_classe in classes.items()
        ]
    )

    registros_duplicidades = _registrar_duplicidades(hashes_sha256, "sha256")
    registros_duplicidades += _registrar_duplicidades(hashes_perceptuais, "dhash_igual")
    df_duplicidades = pd.DataFrame(registros_duplicidades, columns=COLUNAS_DUPLICIDADES)

    if df_caixas.empty:
        df_estatisticas_caixas = pd.DataFrame()
    else:
        df_estatisticas_caixas = (
            df_caixas.groupby(["split", "classe"])[
                ["largura_relativa", "altura_relativa", "area_relativa"]
            ]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
        )
        df_estatisticas_caixas.columns = [
            "_".join(str(parte) for parte in coluna if parte).rstrip("_")
            if isinstance(coluna, tuple)
            else coluna
            for coluna in df_estatisticas_caixas.columns
        ]

    df_resumo.to_csv(pasta_tabelas / "resumo_dataset.csv", index=False)
    df_classes.to_csv(pasta_tabelas / "contagem_classes.csv", index=False)
    df_problemas.to_csv(pasta_tabelas / "problemas_anotacoes.csv", index=False)
    df_duplicidades.to_csv(pasta_tabelas / "duplicidades.csv", index=False)
    df_estatisticas_caixas.to_csv(
        pasta_tabelas / "estatisticas_caixas.csv",
        index=False,
    )
    df_caixas.to_csv(pasta_tabelas / "caixas_validas.csv", index=False)
    df_manifesto.to_csv(pasta_tabelas / "manifesto_imagens.csv", index=False, encoding="utf-8-sig")
    resumo_validacao = {
        "raiz_dataset": str(raiz_dataset.resolve()),
        "imagens": len(df_manifesto),
        "problemas_criticos": int(df_problemas["severidade"].eq("critica").sum()),
        "avisos": int(df_problemas["severidade"].eq("aviso").sum()),
    }
    resumo_validacao["apto_preparacao"] = resumo_validacao["problemas_criticos"] == 0
    (pasta_tabelas / "resumo_validacao.json").write_text(
        json.dumps(resumo_validacao, ensure_ascii=False, indent=2), encoding="utf-8")

    if gerar_figuras:
        _gerar_graficos(df_classes, df_caixas, pasta_figuras)
        _gerar_grade_amostras(df_caixas, pasta_figuras / "amostras_anotadas.jpg", classes)

    resultados = {
        "resumo": df_resumo,
        "classes": df_classes,
        "problemas": df_problemas,
        "duplicidades": df_duplicidades,
        "estatisticas_caixas": df_estatisticas_caixas,
        "caixas": df_caixas,
        "manifesto": df_manifesto,
    }
    if bloquear_erros:
        validar_resultados_auditoria(resultados)
    return resultados


def _gerar_graficos(
    df_classes: pd.DataFrame,
    df_caixas: pd.DataFrame,
    pasta_figuras: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=df_classes.sort_values("instancias", ascending=False),
        x="instancias",
        y="classe",
        color="#3973ac",
    )
    plt.title("Quantidade de instâncias por classe")
    plt.xlabel("Instâncias")
    plt.ylabel("Classe")
    plt.tight_layout()
    plt.savefig(pasta_figuras / "distribuicao_instancias.png", dpi=160)
    plt.close()

    if not df_caixas.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_caixas, x="area_relativa", bins=40, hue="split")
        plt.title("Distribuição da área relativa das caixas")
        plt.xlabel("Área da caixa / área da imagem")
        plt.ylabel("Quantidade")
        plt.tight_layout()
        plt.savefig(pasta_figuras / "distribuicao_area_caixas.png", dpi=160)
        plt.close()


def _gerar_grade_amostras(
    df_caixas: pd.DataFrame,
    caminho_saida: Path,
    classes: dict[int, str],
) -> None:
    if df_caixas.empty:
        return

    amostras: list[tuple[str, Path]] = []
    for nome_classe in classes.values():
        candidatos = df_caixas.loc[df_caixas["classe"] == nome_classe, "imagem"]
        if not candidatos.empty:
            amostras.append((nome_classe, Path(candidatos.iloc[0])))

    if not amostras:
        return

    largura_celula, altura_celula = 420, 320
    colunas = 3
    linhas = (len(amostras) + colunas - 1) // colunas
    grade = Image.new("RGB", (colunas * largura_celula, linhas * altura_celula), "white")
    fonte = ImageFont.load_default()

    for indice, (classe_destaque, caminho_imagem) in enumerate(amostras):
        with Image.open(caminho_imagem) as imagem_original:
            imagem = imagem_original.convert("RGB")

        caixas_imagem = df_caixas[df_caixas["imagem"] == str(caminho_imagem)]
        desenho = ImageDraw.Draw(imagem)

        for caixa in caixas_imagem.to_dict("records"):
            largura, altura = imagem.size
            x1 = (caixa["centro_x"] - caixa["largura_relativa"] / 2) * largura
            y1 = (caixa["centro_y"] - caixa["altura_relativa"] / 2) * altura
            x2 = (caixa["centro_x"] + caixa["largura_relativa"] / 2) * largura
            y2 = (caixa["centro_y"] + caixa["altura_relativa"] / 2) * altura
            cor = "#e63946" if caixa["classe"] == classe_destaque else "#2a9d8f"
            desenho.rectangle((x1, y1, x2, y2), outline=cor, width=3)
            desenho.text((x1 + 3, max(0, y1 - 14)), caixa["classe"], fill=cor, font=fonte)

        imagem.thumbnail((largura_celula - 20, altura_celula - 40))
        celula = Image.new("RGB", (largura_celula, altura_celula), "white")
        x = (largura_celula - imagem.width) // 2
        y = 25 + (altura_celula - 35 - imagem.height) // 2
        celula.paste(imagem, (x, y))
        ImageDraw.Draw(celula).text((10, 7), classe_destaque, fill="black", font=fonte)

        coluna = indice % colunas
        linha = indice // colunas
        grade.paste(celula, (coluna * largura_celula, linha * altura_celula))

    grade.save(caminho_saida, quality=92)


def limpar_download_incompleto(pasta_raw: str | Path) -> None:
    """Remove somente o arquivo parcial criado por um download interrompido."""
    arquivo_parcial = Path(pasta_raw) / "homeobjects-3K.zip.part"
    if arquivo_parcial.exists():
        arquivo_parcial.unlink()
        print(f"Download parcial removido: {arquivo_parcial}")


def remover_extracao_incompleta(pasta_raw: str | Path) -> None:
    """Remove apenas a pasta de extração para permitir uma nova tentativa."""
    pasta_extracao = Path(pasta_raw) / "homeobjects_extraido"
    if pasta_extracao.exists():
        shutil.rmtree(pasta_extracao)
        print(f"Extração incompleta removida: {pasta_extracao}")
