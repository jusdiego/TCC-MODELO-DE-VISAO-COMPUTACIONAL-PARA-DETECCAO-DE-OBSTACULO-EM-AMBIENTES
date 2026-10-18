# Metodologia e funcionamento

## Objetivo

Investigar um detector de objetos de ambientes internos como componente de uma proposta de apoio à percepção do ambiente por pessoas com deficiência visual. A implementação identifica objetos em imagens RGB e produz mensagens de atenção a partir das caixas detectadas. O estudo avalia o detector e a camada de alertas separadamente.

## Taxonomia

O ramo Home usa 12 classes, nesta ordem fixa: `bed`, `sofa`, `chair`, `table`, `lamp`, `tv`, `laptop`, `wardrobe`, `window`, `door`, `potted plant` e `photo frame`. O arquivo [classes_home.json](../datasets/classes_home.json) preserva essa ordem.

O ramo NYU 22/23 usa seis classes locais: `bed`, `sofa`, `chair`, `table`, `door`, `window`. A ordem de porta/janela difere da taxonomia Home. O [mapeamento explícito](../datasets/mapeamento_nyu_home.csv) evita tratar os índices locais como índices globais.

## Preparação dos dados

As etapas iniciais auditaram imagens e anotações, organizaram os arquivos YOLO e produziram a referência Home. SUN RGB-D foi acrescentado em ajustes finos. Open Images foi preparado em lotes até 100 mil imagens, com uma divisão histórica de 80 mil/20 mil; os ciclos posteriores usaram recortes e protocolos específicos, sem presumir cobertura completa de todas as classes em cada imagem.

Auditorias registraram problemas estruturais, possíveis duplicatas e divergências semânticas. Sinais produzidos por modelos auxiliares constituem triagem automática; não foram tratados como reanotação humana confirmada. A investigação posterior comparou treino Home, replay, controle de cobertura de `wardrobe`, limiares e proporções de fontes.

O ramo NYU extraiu caixas das instâncias visíveis das máscaras existentes, sem usar profundidade como entrada do detector. Sua divisão de desenvolvimento foi feita por cenas, preservando a reserva oficial. O ramo final RF-DETR usou exclusivamente o Home no treinamento: 1.818 imagens, além de 456 de validação; a avaliação histórica usou 403 imagens.

## Detecção e alertas

Os detectores retornam classe, caixa e escore de confiança. A política V2 utiliza classe, posição e área relativa na imagem para compor região e prioridade da mensagem. As seis classes do desfecho secundário são `bed`, `sofa`, `chair`, `table`, `wardrobe` e `potted plant`, IDs `[0,1,2,3,7,10]` no Home.

A política não estima distância em metros, trajeto livre ou probabilidade real de colisão. Uma mensagem com prioridade alta representa a regra geométrica implementada, não a confirmação de um risco físico.

## Métricas

Na avaliação principal do 24, as previsões acima do limiar são associadas aos objetos anotados por classe e IoU, conforme o avaliador arquivado em [home24_eval.py](../src/experimentos/24/home24_eval.py) e seu associador. Objetos associados são TP; objetos anotados sem associação são FN; previsões avaliáveis sem associação são FP. As contagens são somadas antes de calcular as métricas micro:

- Precisão: `TP / (TP + FP)`.
- Recall: `TP / (TP + FN)`.
- F1: `2 × TP / (2 × TP + FP + FN)`.

A meta é atingir precisão ≥ 0,80 e recall ≥ 0,70 simultaneamente, com IoU ≥ 0,50. A confiança mínima é um filtro de previsões; confiança 0,50 não significa precisão de 50%. mAP e métricas de alertas são medidas distintas.

Nos ciclos 11–20, diversas tabelas têm como desfecho **alertas** e cobertura parcial por classe. Sua precisão/recall não deve ser incorporada a uma única curva com a precisão/recall micro do detector de 12 classes do 24.

## Seleção no experimento final

RF-DETR Small partiu de pesos pré-treinados em COCO, com resolução 512, EMA e precisão mista FP16. O melhor checkpoint foi selecionado por mAP50–95 no VAL. A confiança foi escolhida no VAL pela regra `min(P/0.8, R/0.7) + 1e-6 × F1`, com desempate em favor da menor confiança na grade fixada.

Os pesos e limiares das três configurações comparadas foram registrados em [decisao_antes_test_24.json](../resultados/experimentos/24/decisao_antes_test_24.json) antes da abertura do TEST nesta rodada. Esse congelamento não apaga a exposição histórica do conjunto, explicitada no [protocolo](../resultados/experimentos/24/protocolo_24.json).

Os intervalos Home foram produzidos com 2.000 réplicas de bootstrap por imagem, incluindo diferenças pareadas. A ausência de grupos de cenas Home validados limita a interpretação desses intervalos. No diagnóstico NYU do 24, o peso e o limiar permaneceram fixos, sem treinamento ou ajuste nessa base.
