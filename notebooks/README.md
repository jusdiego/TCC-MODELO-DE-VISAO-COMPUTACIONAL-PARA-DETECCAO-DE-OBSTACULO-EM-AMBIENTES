# Notebooks principais

A ordem abaixo apresenta a investigação. Cada notebook contém sua configuração e suas etapas; os resultados arquivados são localizados pelo índice e pela documentação dos experimentos.

| Etapa | Notebook | Tema | Ambiente | Saídas gravadas |
| --- | --- | --- | --- | ---: |
| 00 | [00_poc_analise_datasets.ipynb](00_poc_analise_datasets.ipynb) | Exploração inicial das bases | Jupyter/local | 237 |
| 01 | [01_auditoria_homeobjects.ipynb](01_auditoria_homeobjects.ipynb) | Auditoria do HomeObjects | Jupyter/local | 14 |
| 02 | [02_preparacao_dataset.ipynb](02_preparacao_dataset.ipynb) | Preparação dos dados e proteção do TEST | Jupyter/local | 6 |
| 03 | [03_treinamento_yolov8n_colab.ipynb](03_treinamento_yolov8n_colab.ipynb) | Modelo de referência YOLOv8n | Colab | 20 |
| 04 | [04_avaliacao_risco_v1.ipynb](04_avaliacao_risco_v1.ipynb) | Política de alertas V1/V2 | Jupyter/local | 11 |
| 04 | [04_avaliacao_risco_v2_ajustado.ipynb](04_avaliacao_risco_v2_ajustado.ipynb) | Política de alertas V1/V2 | Jupyter/local | 18 |
| 05 | [05_validacao_risco.ipynb](05_validacao_risco.ipynb) | Validação histórica do módulo de risco | Jupyter/local | 26 |
| 05b | [05b_complemento_validacao_risco.ipynb](05b_complemento_validacao_risco.ipynb) | Complemento da validação histórica | Jupyter/local | 14 |
| 05c | [05c_validacao_risco_final.ipynb](05c_validacao_risco_final.ipynb) | Consolidação da validação histórica | Jupyter/local | 21 |
| 05d | [05d_avaliacao_automatica_alertas.ipynb](05d_avaliacao_automatica_alertas.ipynb) | Avaliação automática de detecção e alertas | Jupyter/local | 13 |
| 06 | [06_preparacao_sunrgbd.ipynb](06_preparacao_sunrgbd.ipynb) | Preparação SUN RGB-D | Jupyter/local | 12 |
| 07 | [07_ajuste_fino_sunrgbd_colab.ipynb](07_ajuste_fino_sunrgbd_colab.ipynb) | Ajuste fino com Home e SUN RGB-D | Colab | 12 |
| 07b | [07b_ajuste_fino_sunrgbd_lr_menor_colab.ipynb](07b_ajuste_fino_sunrgbd_lr_menor_colab.ipynb) | Controle de taxa de aprendizado | Colab | 13 |
| 08 | [08_openimages_100k_preparacao_colab.ipynb](08_openimages_100k_preparacao_colab.ipynb) | Preparação Open Images 100 mil | Colab | 24 |
| 08b | [08b_completar_openimages_100k_colab.ipynb](08b_completar_openimages_100k_colab.ipynb) | Complementação Open Images no Colab | Colab | 0 |
| 08c | [08c_completar_openimages_100k_kaggle.ipynb](08c_completar_openimages_100k_kaggle.ipynb) | Complementação Open Images no Kaggle | Kaggle | 0 |
| 09 | [09_treino_openimages_piloto_kaggle.ipynb](09_treino_openimages_piloto_kaggle.ipynb) | Piloto Open Images e supervisão parcial | Kaggle | 26 |
| 10 | [10_treino_openimages_100k_kaggle.ipynb](10_treino_openimages_100k_kaggle.ipynb) | Treino ampliado Open Images 80/20 | Kaggle | 40 |
| 11 | [11_diagnostico_validacao_independente_kaggle.ipynb](11_diagnostico_validacao_independente_kaggle.ipynb) | Diagnóstico e validação externa | Kaggle | 0 |
| 12 | [12_ajuste_home_replay_controlado_kaggle.ipynb](12_ajuste_home_replay_controlado_kaggle.ipynb) | Home versus replay controlado | Kaggle | 0 |
| 13 | [13_auditoria_rotulos_erros_kaggle.ipynb](13_auditoria_rotulos_erros_kaggle.ipynb) | Auditoria dos rótulos e erros | Kaggle | 0 |
| 14 | [14_diagnostico_confianca_automatico_kaggle.ipynb](14_diagnostico_confianca_automatico_kaggle.ipynb) | Análise do limiar global de confiança | Kaggle | 0 |
| 15 | [15_auditoria_automatica_treino_kaggle.ipynb](15_auditoria_automatica_treino_kaggle.ipynb) | Triagem automática do treino | Kaggle | 0 |
| 16 | [16_controle_wardrobe_por_fonte_kaggle.ipynb](16_controle_wardrobe_por_fonte_kaggle.ipynb) | Controle da cobertura de wardrobe por fonte | Kaggle | 0 |
| 17 | [17_limiares_por_classe_kaggle.ipynb](17_limiares_por_classe_kaggle.ipynb) | Limiares de confiança por classe | Kaggle | 0 |
| 18 | [18_equilibrio_fontes_kaggle.ipynb](18_equilibrio_fontes_kaggle.ipynb) | Equilíbrio das fontes de treino | Kaggle | 0 |
| 19 | [19_consolidacao_demonstracao_kaggle.ipynb](19_consolidacao_demonstracao_kaggle.ipynb) | Consolidação e demonstração | Kaggle | 0 |
| 20 | [20_avaliacao_final_reserva_kaggle.ipynb](20_avaliacao_final_reserva_kaggle.ipynb) | Avaliação da reserva externa | Kaggle | 0 |
| 21 | [21_yolov8s_treinamento_ampliado_kaggle.ipynb](21_yolov8s_treinamento_ampliado_kaggle.ipynb) | Treinamento ampliado YOLOv8s | Kaggle | 0 |
| 22 | [22_nyu_compacto_preparacao_windows.ipynb](22_nyu_compacto_preparacao_windows.ipynb) | Preparação compacta NYU | Windows | 0 |
| 23 | [23_nyu_compacto_treinamento_colab.ipynb](23_nyu_compacto_treinamento_colab.ipynb) | Treinamento e teste NYU | Colab | 0 |
| 24 | [24_rfdetr_small_home_colab.ipynb](24_rfdetr_small_home_colab.ipynb) | RF-DETR Small: Home e diagnóstico NYU | Colab | 0 |

O notebook 24 Colab foi recomposto a partir do código arquivado da execução, sem saídas fabricadas. Os resultados finais são os do pacote 24. A variante Kaggle corrigida permanece em `historico/24/`.

[Hipóteses e conclusões](../docs/02_experimentos.md) · [Índice completo com hashes e origens](../docs/indice_notebooks.csv) · [Reprodução](../docs/04_reproducao.md)
