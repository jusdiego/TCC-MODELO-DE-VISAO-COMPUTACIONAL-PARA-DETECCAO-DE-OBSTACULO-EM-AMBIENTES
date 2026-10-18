# Ambientes de execução

Não existe um ambiente único validado para todos os notebooks. A sequência inclui Windows/Jupyter, Colab e Kaggle, com versões fixadas em etapas posteriores.

| Ciclo | Referência |
| --- | --- |
| Etapas iniciais | `requirements.txt`, scripts Windows e células do notebook. O arquivo da raiz é o conjunto inicial preservado, não um lock de toda a pesquisa. |
| Controles 11–20 | Células de instalação e configurações/ambientes dentro dos respectivos resultados. |
| 21 | [configuracao_treinamento_21.json](../resultados/experimentos/21/configuracao_treinamento_21.json). Treino YOLOv8s em FP32, `amp=false`, resolução 768; estado recuperado pausado/retomável. |
| 23 | [configuracao_nyu23.json](../resultados/experimentos/23/configuracao_nyu23.json). YOLOv8s, resolução 640, `amp=false`, Ultralytics 8.4.150. |
| 24 efetivamente executado | [ambiente_24.json](../resultados/experimentos/24/ambiente_24.json): Colab, Python 3.13.15, Tesla T4, PyTorch 2.11.0+cu130, CUDA 13.0. |

[requirements_rf24_registrado.txt](requirements_rf24_registrado.txt) registra versões observadas dos pacotes principais do 24. É uma referência de ambiente, não um lock de todas as dependências transitivas. As wheels PyTorch/CUDA e o runtime precisam ser compatíveis com a máquina; o notebook orienta a instalação/verificação.

Os scripts de organização e conferência em `scripts/preparar_home24.py`, `scripts/verificar_pacotes.py` e `scripts/verificar_repositorio.py` utilizam apenas a biblioteca padrão Python 3.10+.
