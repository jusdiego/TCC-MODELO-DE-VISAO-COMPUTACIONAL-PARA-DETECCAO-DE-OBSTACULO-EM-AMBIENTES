"""Parâmetros e funções da V2 original, sem reestimar com o TEST."""
import math

NOMES_PT = {"bed": "cama", "sofa": "sofá", "chair": "cadeira", "table": "mesa",
            "wardrobe": "guarda-roupa", "potted plant": "planta em vaso"}
VERBOS_PT = {"bed": "detectada", "sofa": "detectado", "chair": "detectada", "table": "detectada",
             "wardrobe": "detectado", "potted plant": "detectada"}
ORDEM_PRIORIDADE = {"sem_prioridade": 0, "baixo": 1, "medio": 2, "alto": 3}


def avaliar_bbox(classe, bbox, largura, altura, versao="v2"):
    if versao not in {"v1", "v2"}:
        raise ValueError("Versão deve ser v1 ou v2.")
    x1, y1, x2, y2 = map(float, bbox)
    if largura <= 0 or altura <= 0 or not all(math.isfinite(v) for v in [x1, y1, x2, y2]):
        raise ValueError("Dimensões ou coordenadas inválidas.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Caixa sem área positiva.")
    posicao = classificar_posicao((x1 + x2) / 2, largura)
    tamanho, area = classificar_tamanho((x2 - x1) * (y2 - y1), largura * altura)
    vertical, base = classificar_posicao_vertical(y2, altura)
    resultado = (classificar_risco_v2(classe, posicao, tamanho, vertical) if versao == "v2"
                 else classificar_risco_v1(classe, posicao, tamanho))
    nivel = resultado["nivel_risco"]
    return {"prioridade": nivel if nivel in {"baixo", "medio", "alto"} else "sem_prioridade",
            "status_original": nivel, "pontuacao": resultado["pontuacao"],
            "posicao": posicao, "tamanho": tamanho, "regiao_vertical": vertical,
            "area_relativa": area, "base_vertical_relativa": base,
            "ajuste_vertical": resultado.get("ajuste_vertical", 0)}


def gerar_mensagem(objeto, nivel_minimo="medio"):
    if nivel_minimo not in {"medio", "alto"}:
        raise ValueError("Nível mínimo deve ser medio ou alto.")
    prioridade = objeto["prioridade"]
    if ORDEM_PRIORIDADE.get(prioridade, 0) < ORDEM_PRIORIDADE[nivel_minimo]:
        return ""
    posicao = {"centro": "central", "esquerda": "esquerda", "direita": "direita"}[objeto["posicao"]]
    classe = objeto["classe"]
    return (f"Atenção: {NOMES_PT[classe]} {VERBOS_PT[classe]} na região {posicao} da imagem. "
            f"Prioridade {'alta' if prioridade == 'alto' else 'média'}.")


def gerar_mensagens_tabela(df, coluna="nivel_risco_v2"):
    if df.empty:
        return []
    candidatos = df.loc[df[coluna].isin(["medio", "alto"])].copy()
    candidatos["_ordem"] = candidatos[coluna].map(ORDEM_PRIORIDADE)
    candidatos = candidatos.sort_values(["_ordem", "ocupacao_imagem"], ascending=[False, False])
    return [gerar_mensagem({"classe": r["classe"], "posicao": r["posicao"], "prioridade": r[coluna]})
            for _, r in candidatos.iterrows()]

PESO_OBJETO = {
    "bed": 3,
    "sofa": 3,
    "chair": 3,
    "table": 3,
    "wardrobe": 3,
    "potted plant": 2,
}

CLASSES_CONTEXTUAIS = {
    "lamp",
    "door",
}

CLASSES_IGNORADAS = {
    "tv",
    "laptop",
    "window",
    "photo frame",
}


LIMITE_ESQUERDA = 1 / 3
LIMITE_DIREITA = 2 / 3

LIMITE_AREA_MEDIA = 0.05
LIMITE_AREA_GRANDE = 0.20

LIMITE_VERTICAL_SUPERIOR = 0.60
LIMITE_VERTICAL_INFERIOR = 0.80

PESO_POSICAO = {
    "esquerda": 1,
    "centro": 3,
    "direita": 1,
}

PESO_TAMANHO = {
    "pequeno": 1,
    "medio": 2,
    "grande": 3,
}

AJUSTE_VERTICAL = {
    "superior": -2,
    "intermediaria": -1,
    "inferior": 0,
}

def classificar_posicao(
    centro_x: float,
    largura_imagem: int,
) -> str:
    centro_normalizado = centro_x / largura_imagem

    if centro_normalizado < LIMITE_ESQUERDA:
        return "esquerda"

    if centro_normalizado < LIMITE_DIREITA:
        return "centro"

    return "direita"


def classificar_tamanho(
    area_box: float,
    area_imagem: float,
):
    proporcao = area_box / area_imagem

    if proporcao < LIMITE_AREA_MEDIA:
        return "pequeno", proporcao

    if proporcao < LIMITE_AREA_GRANDE:
        return "medio", proporcao

    return "grande", proporcao


def classificar_posicao_vertical(
    y2: float,
    altura_imagem: int,
):
    proporcao_vertical = y2 / altura_imagem

    if proporcao_vertical < LIMITE_VERTICAL_SUPERIOR:
        return "superior", proporcao_vertical

    if proporcao_vertical < LIMITE_VERTICAL_INFERIOR:
        return "intermediaria", proporcao_vertical

    return "inferior", proporcao_vertical

def classificar_risco_v1(
    classe: str,
    posicao: str,
    tamanho: str,
):
    if classe in CLASSES_IGNORADAS:
        return {
            "pontuacao": 0,
            "nivel_risco": "ignorado",
            "motivo": "classe sem relevância direta para locomoção",
        }

    if classe in CLASSES_CONTEXTUAIS:
        return {
            "pontuacao": None,
            "nivel_risco": "contextual",
            "motivo": "classe exige informação adicional de contexto",
        }

    if classe not in PESO_OBJETO:
        return {
            "pontuacao": None,
            "nivel_risco": "nao_mapeado",
            "motivo": "classe não configurada",
        }

    pontuacao = (
        PESO_OBJETO[classe]
        + PESO_POSICAO[posicao]
        + PESO_TAMANHO[tamanho]
    )

    if pontuacao <= 4:
        nivel = "baixo"
    elif pontuacao <= 6:
        nivel = "medio"
    else:
        nivel = "alto"

    return {
        "pontuacao": pontuacao,
        "nivel_risco": nivel,
        "motivo": (
            f"{classe} + posição {posicao} + tamanho {tamanho}"
        ),
    }

def classificar_risco_v2(
    classe: str,
    posicao: str,
    tamanho: str,
    posicao_vertical: str,
):
    avaliacao_v1 = classificar_risco_v1(
        classe=classe,
        posicao=posicao,
        tamanho=tamanho,
    )

    if avaliacao_v1["nivel_risco"] in {
        "ignorado",
        "contextual",
        "nao_mapeado",
    }:
        return {
            "pontuacao": avaliacao_v1["pontuacao"],
            "nivel_risco": avaliacao_v1["nivel_risco"],
            "ajuste_vertical": 0,
            "motivo": avaliacao_v1["motivo"],
        }

    ajuste_vertical = AJUSTE_VERTICAL[posicao_vertical]

    pontuacao = max(
        1,
        avaliacao_v1["pontuacao"] + ajuste_vertical,
    )

    if pontuacao <= 4:
        nivel = "baixo"
    elif pontuacao <= 6:
        nivel = "medio"
    else:
        nivel = "alto"

    return {
        "pontuacao": pontuacao,
        "nivel_risco": nivel,
        "ajuste_vertical": ajuste_vertical,
        "motivo": (
            f"{classe} + posição {posicao} "
            f"+ tamanho {tamanho} "
            f"+ região vertical {posicao_vertical}"
        ),
    }
