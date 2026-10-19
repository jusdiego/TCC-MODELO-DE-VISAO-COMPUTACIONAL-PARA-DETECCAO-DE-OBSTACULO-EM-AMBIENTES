# Como consultar, conferir e reproduzir

## 1. Ler os resultados sem repetir treinamento

Começar pelo [README](../README.md), pela [análise](03_resultados.md) e pela [sequência de experimentos](02_experimentos.md). Os CSVs abrem no GitHub ou em planilhas; os JSONs contêm os registros completos correspondentes.

O pacote `TCC_Resultados_Para_Apresentacao.zip`, destinado ao Release, reúne relatórios HTML, galerias e previsões dos experimentos recuperados. Abrir esses relatórios não exige GPU ou imagens das bases completas.

## 2. Conferir esta cópia

Com Python 3.10 ou superior, na raiz do repositório:

```powershell
python scripts/verificar_repositorio.py
```

O script utiliza a biblioteca padrão: confere os hashes de distribuição, a origem das cópias, a estrutura/contagem dos notebooks, os tamanhos das partições e a coerência entre contagens por imagem, classe e métricas finais. Não executa células, downloads ou inferências.

## 3. Recuperar as bases

Abrir [ACESSO_A_BASE.txt](../ACESSO_A_BASE.txt) para os quatro links de download e entrar com uma conta institucional autorizada pelo Centro Universitário Impacta. Os links foram configurados como Leitor para pessoas da instituição com o link; o repositório público não torna esses downloads anônimos.

Consultar também [datasets/README.md](../datasets/README.md). O inventário informa nomes, hashes e caminhos históricos. Cada notebook exige as entradas de sua etapa; as imagens completas ficam no Drive.

Para conferir uma pasta com os ZIPs originais:

```powershell
python scripts/verificar_pacotes.py --origem "C:\caminho\TCC_Visao_Computacional"
```

Para criar o pacote portátil do Home 24 com a divisão já fixada:

```powershell
python scripts/preparar_home24.py --origem "C:\caminho\homeobjects_sunrgbd_validado.zip"
```

Esses scripts precisam de espaço em disco para os arquivos produzidos, mas não precisam de GPU. O pacote portátil é uma nova embalagem da mesma seleção; os notebooks históricos ainda conferem o ZIP fonte original.

## 4. Escolher a execução pretendida

| Objetivo | Material de partida | Entradas adicionais |
| --- | --- | --- |
| Estudar a investigação inicial | Notebooks 00–10 | Base e configurações da etapa, dependências iniciais. |
| Reproduzir auditorias/controles | Notebooks 11–20 | Pacotes anteriores autenticados, pesos e dados dos protocolos. |
| Retomar o estado recuperado do 21 | Notebook 21 e pacote original | Estado de treino/otimizador e base preparada; um peso de inferência sozinho não equivale a retomada. |
| Reproduzir o ramo compacto NYU | Notebook 22 e depois 23 | MAT/split oficiais ou os ZIPs preparados, referência `base_nyu22.json`. |
| Reproduzir o experimento final | Notebook 24 | ZIP Home histórico, pré-treino RF-DETR, entradas/pesos de referência para comparação e, opcionalmente, reserva NYU. |

Para o 24 no Kaggle, usar a [variante Kaggle corrigida](../historico/24/24_rfdetr_small_home_kaggle.ipynb). A execução que gerou os resultados finais ocorreu em Colab/Tesla T4. O [notebook Colab](../notebooks/24_rfdetr_small_home_colab.ipynb) foi recomposto a partir do código arquivado da execução, conforme a proveniência documentada.

## 5. Ambiente e caminhos

As versões registradas estão em [ambientes/](../ambientes/README.md). O `requirements.txt` da raiz pertence aos ciclos iniciais. Os notebooks posteriores instalam/fixam suas próprias versões e preparam código embutido. Não instalar todos os conjuntos de dependências em um único ambiente presumindo compatibilidade.

As configurações e células ainda identificam os ambientes históricos: Windows, Drive, Colab e Kaggle. Escolher a variante correspondente e ajustar somente a raiz de trabalho/entrada quando necessário. Caminhos de resultados antigos permanecem nos registros como proveniência, sem indicar que existam em outra máquina.

Os módulos em `src/experimentos/` preservam sua organização original de importações. A execução orientada é pelo notebook correspondente, que instala os módulos e verifica as fontes. Não há um comando único que reexecute toda a investigação de uma vez.

## 6. Integridade da avaliação

Preservar classes, partições, filtros, resolução, regra de associação, limiar e peso registrados. Uma nova execução com outras versões, dados ou pós-processamento é uma reprodução com alterações, que precisa de identificação própria. O TEST Home continua histórico; seu uso repetido não produz novas confirmações independentes.

Para conferir o resultado arquivado, não é necessário treinar. Para uma comparação por inferência das três configurações, além do RF-DETR são necessários candidato 10 e 21/época 21. Seus hashes estão em [decisao_antes_test_24.json](../resultados/experimentos/24/decisao_antes_test_24.json). O pacote de apresentação contém o peso final RF-DETR, não todos os checkpoints históricos.
