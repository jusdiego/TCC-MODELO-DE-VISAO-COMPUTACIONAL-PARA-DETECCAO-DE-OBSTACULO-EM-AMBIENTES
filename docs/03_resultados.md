# Resultados obtidos e interpretação

## Comparação final no Home TEST histórico

Avaliação de 403 imagens, 12 classes, IoU ≥ 0,50 e confiança 0,50 para as três configurações, escolhida no VAL pelo protocolo do 24. As métricas são micro do detector.

| Configuração | TP | FN | FP | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n — candidato 10 | 2189 | 1276 | 611 | 78,18% | 63,17% | 69,88% |
| YOLOv8s — 21/época 21 | 2273 | 1192 | 610 | 78,84% | 65,60% | 71,61% |
| RF-DETR Small — 24 | 2474 | 991 | 549 | 81,84% | 71,40% | 76,26% |

Fonte: [comparacao_24.csv](../resultados/experimentos/24/comparacao_24.csv). O RF-DETR recuperou mais 285 objetos e produziu 62 falsos positivos a menos que o candidato 10. As diferenças foram +3.66 pontos percentuais em precisão, +8.23 em recall e +6.38 em F1.

Essa é uma comparação de configurações completas, com diferenças de dados, resolução e receita. Não é uma ablação controlada do efeito da arquitetura.

## Incerteza e condição histórica

| Métrica RF-DETR | Estimativa | IC de 95% |
| --- | ---: | ---: |
| Precisão | 81,84% | 80,06%–83,64% |
| Recall | 71,40% | 69,46%–73,50% |
| F1 | 76,26% | 74,65%–77,92% |

Fonte: [bootstrap pareado](../resultados/experimentos/24/bootstrap_test_historico_24.json), 2.000 réplicas por imagem. A meta foi atingida pela estimativa central; o intervalo de recall cruza o limite de 70%. O agrupamento por cenas Home não foi validado. O TEST já havia sido consultado anteriormente, conforme [protocolo](../resultados/experimentos/24/protocolo_24.json). Esses pontos limitam a confirmação independente.

## Detector secundário e alertas

As seis classes secundárias são `bed`, `sofa`, `chair`, `table`, `wardrobe` e `potted plant`, definidas pela política V2. No mesmo limiar principal, o detector nessas classes apresentou P 83,86%, R 76,67% e F1 80,10%.

Os **alertas V2** apresentaram P 84,28%, R 80,54% e F1 82,37%. Entre 458 objetos de prioridade alta segundo a referência heurística, 54 ficaram sem alerta, contra 89 no candidato 10 e 74 no 21/época 21. Esses números avaliam a regra de mensagens, não segurança física de locomoção.

## Desempenho por classe

| Classe | GT | TP | FN | FP | Precisão | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bed | 22 | 19 | 3 | 3 | 86,36% | 86,36% |
| sofa | 397 | 329 | 68 | 36 | 90,14% | 82,87% |
| chair | 305 | 252 | 53 | 62 | 80,25% | 82,62% |
| table | 467 | 373 | 94 | 78 | 82,71% | 79,87% |
| lamp | 304 | 168 | 136 | 84 | 66,67% | 55,26% |
| tv | 54 | 42 | 12 | 8 | 84,00% | 77,78% |
| laptop | 4 | 4 | 0 | 2 | 66,67% | 100,00% |
| wardrobe | 109 | 51 | 58 | 22 | 69,86% | 46,79% |
| window | 371 | 158 | 213 | 88 | 64,23% | 42,59% |
| door | 85 | 48 | 37 | 24 | 66,67% | 56,47% |
| potted plant | 787 | 576 | 211 | 107 | 84,33% | 73,19% |
| photo frame | 560 | 454 | 106 | 35 | 92,84% | 81,07% |

Fonte: [classes_test_historico_24.csv](../resultados/experimentos/24/classes_test_historico_24.csv). As diferenças entre classes e o baixo suporte de algumas categorias devem acompanhar o agregado. `laptop`, por exemplo, tem somente quatro objetos de referência.

## Diagnóstico externo no NYU

O peso e o limiar do RF-DETR permaneceram fixos. Nas 654 imagens e seis classes comuns (`bed`, `sofa`, `chair`, `table`, `window`, `door`), houve TP 966, FN 1162 e FP 398: P **70,82%**, R **45,39%**, F1 **55,33%**. Nenhuma imagem foi excluída por sobreposição nessa execução.

Fonte: [avaliacao_externa_24.json](../resultados/experimentos/24/avaliacao_externa_24.json). O resultado mostra limitação de transferência. O NYU usa caixas derivadas das máscaras visíveis e outra composição de classes; não comparar seu percentual como se fosse o mesmo benchmark Home.

## Resultados anteriores que precisam permanecer no relato

Os controles 12, 14, 16, 17 e 18 não satisfizeram seus critérios de seleção. As auditorias 13/15 produziram sinais e registros, sem uma base reanotada por humanos. O 21 estava pausado/retomável após 24 épocas. O 23 terminou com P 54,34%, R 50,28% e F1 52,23% no NYU reservado, confiança 0,25; seu melhor checkpoint foi a época 41.

O notebook 20 avaliou 300 imagens da reserva Open Images, com **métricas de alertas sob cobertura parcial**: baseline P 52,51%/R 52,04%, candidato 10 P 57,26%/R 60,63%. Esses resultados não são o mesmo estimando das 12 classes do detector no 24. Consulte [tabela do 20](../resultados/experimentos/20/comparacao_final_20.csv).

## Treinamento final e correção de execução

O RF-DETR 24 terminou na época 25 por parada antecipada, melhor checkpoint EMA na época 13. O registro contabiliza 83.35 minutos de treinamento. Seu ambiente efetivo foi Colab/Tesla T4; há uma variante Kaggle preparada.

A avaliação foi concluída após a correção do índice de background. Apenas o índice 12 confirmado como background foi filtrado; o registro contém 55 ocorrências, entre VAL, TEST e NYU. O arquivo histórico `falha_24.json` preserva o erro anterior; o [resumo final](../resultados/experimentos/24/resumo_experimento_24.json) registra `completed=true`. Essa correção não exigiu um novo treinamento.

## Arquivos para a apresentação

Priorizar a comparação Home, o desempenho por classe, os intervalos, o diagnóstico NYU e a sequência de hipóteses em [experimentos](02_experimentos.md). Os relatórios HTML e a demonstração antiga estão no pacote completo de resultados destinado ao Release. A demonstração 19 apresenta os modelos daquele ciclo; não deve ser atribuída ao RF-DETR 24.
