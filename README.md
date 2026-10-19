# Visão computacional para acessibilidade em ambientes internos

Material de desenvolvimento e avaliação do TCC **Modelo de visão computacional para detecção de obstáculos em ambientes internos aplicado à acessibilidade de pessoas com deficiência visual**, do Bacharelado em Ciência da Computação da Faculdade Impacta Tecnologia, 2026.

O projeto investiga a detecção de objetos domésticos e a geração de mensagens de atenção a partir de sua classe, posição e área na imagem. A investigação percorreu a preparação de bases, modelos YOLOv8, auditorias de dados e avaliação, ajustes de treinamento e de confiança, um ramo NYU e um experimento final com RF-DETR Small.

## Resultado principal

Meta do detector: **precisão micro ≥ 80% e recall micro ≥ 70%, simultaneamente**, com classe correta e IoU ≥ 0,50. O experimento 24 atingiu essa meta na estimativa pontual do **TEST Home histórico**, com confiança mínima 0,50 escolhida no VAL.

| Modelo/configuração | Conjunto | Imagens | Precisão | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n — candidato 10 | Home TEST histórico, 12 classes | 403 | 78,18% | 63,17% | 69,88% |
| YOLOv8s — 21, época 21 | Home TEST histórico, 12 classes | 403 | 78,84% | 65,60% | 71,61% |
| **RF-DETR Small — 24** | **Home TEST histórico, 12 classes** | **403** | **81,84%** | **71,40%** | **76,26%** |
| RF-DETR Small — diagnóstico externo | NYU TEST, 6 classes em comum | 654 | 70,82% | 45,39% | 55,33% |

As três primeiras linhas foram avaliadas no protocolo do notebook 24, com limiares escolhidos no mesmo VAL. A linha NYU descreve outra base e outra composição de classes. O resultado Home não constitui confirmação em teste novo: esse conjunto já havia sido consultado no projeto. Seu intervalo de 95% para recall é **69,46%–73,50%**. As configurações diferem em arquitetura, resolução e treinamento; a comparação não isola o efeito da arquitetura.

Fontes: [comparação Home](resultados/experimentos/24/comparacao_24.csv), [avaliação NYU](resultados/experimentos/24/avaliacao_externa_24.json) e [bootstrap](resultados/experimentos/24/bootstrap_test_historico_24.json). Consulte a [análise completa](docs/03_resultados.md).

A coluna mAP50 dos três modelos foi reconstruída das previsões arquivadas. Consulte o [script de reprodução](scripts/reconstruir_map50.py) e a [documentação do cálculo](docs/09_reconstrucao_map50.md).

## Navegação

| Material | Onde consultar |
| --- | --- |
| Problema, funcionamento e avaliação | [Metodologia](docs/01_metodologia.md) |
| Hipóteses e conclusões de cada etapa | [Histórico dos experimentos](docs/02_experimentos.md) |
| Sequência dos 32 notebooks principais | [Índice comentado](notebooks/README.md) |
| 39 variantes e adaptações de ambiente | [Histórico de notebooks](historico/README.md) |
| Bases, divisões, classes e arquivos necessários | [Bases de dados](datasets/README.md) |
| Links e instruções para baixar as bases | [Acesso às bases](ACESSO_A_BASE.txt) |
| Como conferir ou reproduzir o trabalho | [Reprodução](docs/04_reproducao.md) |
| Alcance das conclusões | [Limitações](docs/05_limitacoes.md) |
| Como publicar o material | [Publicação no GitHub](docs/06_publicacao_github.md) |
| Origem e conferência dos arquivos | [Integridade e proveniência](docs/08_integridade_proveniencia.md) |

## Organização

`notebooks/` contém a sequência principal. `historico/` preserva versões e dependências anteriores. `src/` reúne módulos dos primeiros ciclos e cópias do código arquivado por experimento. `resultados/` contém tabelas e registros de resultados. `datasets/` preserva manifestos, divisões e documentação das bases. `modelos/` registra a identidade do peso final. `publicacao/` descreve os arquivos destinados aos Releases.

Este repositório reúne **32 notebooks principais e 39 variantes**; essa contagem não significa 71 treinamentos. Há notebooks de preparação, auditoria, avaliação, correção e adaptação de ambiente.

## Consulta e reprodução

Para ler os resultados, abra as páginas e tabelas acima; não é necessário treinar novamente. Para conferir hashes, contagens e métricas agregadas com Python 3.10 ou superior:

```bash
python scripts/verificar_repositorio.py
```

Os notebooks de treinamento precisam das bases e dos pesos indicados em cada etapa. Não há um único ambiente de dependências para toda a sequência. Consulte [ambientes registrados](ambientes/README.md) e o [guia de reprodução](docs/04_reproducao.md).

## Bases, pesos e pacotes completos

Os links dos quatro grupos de bases estão em **[ACESSO_A_BASE.txt](ACESSO_A_BASE.txt)**, com nomes, versões, hashes e instruções. Os arquivos no Google Drive estão configurados como **Leitor para contas do Centro Universitário Impacta com o link**; o download exige uma conta institucional autorizada. A conta proprietária não ofereceu compartilhamento para qualquer pessoa com o link.

Os manifestos e as divisões estão versionados em `datasets/`; os ZIPs de imagens são distribuídos pelo Drive. O inventário exato está em [arquivos esperados](datasets/arquivos_esperados.csv). A base do experimento 24 contém **1.818 imagens de treino, 456 de VAL e 403 de TEST histórico**.

O peso final e o arquivo `TCC_Resultados_Para_Apresentacao.zip` estão disponíveis nos [anexos da versão v1.0.0](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES/releases/tag/v1.0.0): [pacote de resultados](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES/releases/download/v1.0.0/TCC_Resultados_Para_Apresentacao.zip) e [peso RF-DETR Small](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES/releases/download/v1.0.0/modelo_rfdetr24.pth). Os tamanhos e hashes estão nos [metadados da publicação](publicacao/arquivos_release.json). O [repositório](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES) é público; os arquivos e anexos podem ser consultados e baixados sem login. Consulte as [instruções de distribuição](publicacao/README.md).

## Autores

- Diego Ramos Justina
- Fernanda Silva Santos
- Guilherme Almeida dos Santos
- João Antônio da Silva Junior
- Samanta Aparecida de Souza

Orientação: Prof. Dr. Gilberto Perez. Faculdade Impacta Tecnologia — Ciência da Computação, 2026. Os dados para citação estão em [CITATION.cff](CITATION.cff).

## Alcance da aplicação

As mensagens V2 são heurísticas sobre caixas 2D. Os resultados documentados não medem distância física nem demonstram segurança de locomoção com usuários. A análise de generalização e das classes com menor desempenho faz parte das [limitações do estudo](docs/05_limitacoes.md).

As fontes de dados e bibliotecas mantêm seus próprios termos e atribuições, documentados em [avisos de terceiros](AVISOS_DE_TERCEIROS.md).
