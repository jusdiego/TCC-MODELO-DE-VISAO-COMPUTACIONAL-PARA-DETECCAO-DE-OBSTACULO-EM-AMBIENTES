# Código por ciclo de investigação

Os módulos na raiz de `src/` pertencem aos ciclos iniciais de auditoria, preparação e alertas. `experimentos/<etapa>/` contém cópias do código arquivado nos respectivos pacotes de resultados, com sua estrutura de importação original. O código 11 está preservado nos notebooks e também em versões de referência arquivadas em etapas posteriores; não foi inventada uma pasta de código original ausente do pacote 11.

A execução principal é feita pelos notebooks, que preparam os módulos e entradas do seu próprio ambiente. Esta pasta não constitui uma API unificada: nomes como `diag11_core` e `home24_common` dependem do diretório preparado em cada etapa. Para reproduzir uma rodada histórica, usar as entradas, os hashes e o bootstrap do notebook correspondente.

O bundle original RF-DETR 24 reúne seis módulos `home24_*` e três arquivos em `src/`, além da correção de background arquivada posteriormente. O notebook Colab recomposto preserva esse bundle. A variante Kaggle inclui um adaptador de transporte próprio.

As cópias de código nesta pasta permanecem sem alteração de lógica; a origem e o SHA-256 de cada arquivo estão em `auditoria/catalogo_origens.csv`.
