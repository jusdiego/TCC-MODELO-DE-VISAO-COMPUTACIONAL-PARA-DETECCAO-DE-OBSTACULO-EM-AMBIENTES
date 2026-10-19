# Reconstrução do mAP50 de teste do Colab

O [script](../scripts/reconstruir_map50.py) reproduz o mAP50 dos três modelos
avaliados no pacote original do experimento 24 Colab. Ele usa as previsões
preservadas e os GTs históricos, sem executar treinamento nem nova inferência.

| Modelo | mAP50 reconstruído | Arredondado |
|---|---:|---:|
| YOLOv8n candidato 10 | 0,6748580838720697 | 67,49% |
| YOLOv8s 21 época 21 | 0,6787673977160485 | 67,88% |
| RF-DETR Small principal | 0,7899370204212569 | 78,99% |

## Entradas e verificações

Baixar o arquivo TCC_Resultados_Para_Apresentacao.zip da
[Release v1.0.0](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES/releases/tag/v1.0.0).
Seu SHA-256 é 5ea3e4b07026a8fbfcd1dd8adfae3755f4f19122b582641c885ea733c660e918.
O pacote_experimento_24.zip original do Colab também é aceito.

O ZIP contém os blocos test_historico/modelo/cache/chave/bloco_NNNN.json.gz.
O script confere o hash de cada bloco e sua identidade. Os GTs são reconstruídos
do CSV histórico avaliacao_objetos.csv já versionado no repositório, incluindo
os objetos que o detector anterior não encontrou.

Os hashes de imagem e rótulo dos manifestos antigo e do 24 precisam coincidir.
O script exige 403 imagens e 3.465 objetos e reproduz, imagem a imagem, os TP,
FN e FP originais de cada modelo no limiar operacional 0,50. Se uma verificação
falhar, encerra com erro.

## Método da AP

As previsões do Colab foram capturadas desde confiança 0,01, até 300 por imagem.
Essas previsões entram integralmente no cálculo de AP50, sem aplicar o corte
operacional 0,50 antes da curva.

Para cada classe, as previsões são ordenadas por confiança decrescente.
Cada previsão pode corresponder a uma anotação da mesma classe, pelo melhor
IoU disponível, desde que seja pelo menos 0,50. Cada anotação é utilizada uma
única vez. A envoltória de precisão é amostrada em 101 níveis de recall de
0 a 1, com precisão zero além do recall máximo alcançado. O mAP50 é a média
das doze APs.

O pareamento para validar o ponto operacional preserva separadamente o
associador histórico do Colab, ordenado por IoU. AP e métricas pontuais têm
procedimentos distintos, documentados no script.

## Execução

Requisitos: Python 3.10 ou superior e numpy 1.24 ou superior.
Na raiz do repositório:

    python scripts/reconstruir_map50.py --pacote-colab "/caminho/TCC_Resultados_Para_Apresentacao.zip" --saida "/caminho/reconstrucao_map50_colab.json"

O script verifica os valores de referência apenas após calcular o resultado;
não altera dados nem ajusta o cálculo para forçar coincidência.
Os [resultados desta execução](../resultados/experimentos/24/reconstrucao_map50_colab.json)
incluem AP por classe, identidades das entradas e contagens reproduzidas.

## Diagnóstico do corte prévio

O parâmetro opcional --corte-previo 0.50 permite verificar o efeito de truncar
as previsões antes da AP:

    python scripts/reconstruir_map50.py --pacote-colab "/caminho/TCC_Resultados_Para_Apresentacao.zip" --corte-previo 0.50 --saida "/caminho/reconstrucao_map50_corte_050.json"

Nesse diagnóstico, os mAP50 são 51,80%, 56,88% e 66,48%, respectivamente.
Esses valores não correspondem à coluna mAP50 do TCC.
O [JSON diagnóstico](../resultados/experimentos/24/reconstrucao_map50_corte_050.json)
registra explicitamente o corte aplicado.

Esta reconstrução se refere ao pacote Colab. Não generalizar seus parâmetros
para outros avaliadores ou execuções.
