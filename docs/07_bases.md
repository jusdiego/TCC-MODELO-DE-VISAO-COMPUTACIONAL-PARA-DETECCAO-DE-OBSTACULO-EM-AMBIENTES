# Bases, versões e arquivos para reprodução

## HomeObjects e o recorte histórico

Fonte pública: [HomeObjects-3K — Ultralytics](https://docs.ultralytics.com/datasets/detect/homeobjects-3k). A documentação oficial informa 12 classes e disponibiliza um ZIP público. A base preparada do projeto foi reorganizada e auditada; os hashes e partições históricos são a referência para os resultados deste TCC.

Para o 24, o [manifesto](../datasets/manifestos/home24.json.gz) identifica 1.818 imagens de treino, 456 de validação e 403 de TEST histórico. As seis exclusões estão em [home24_exclusoes.csv](../datasets/divisoes/home24_exclusoes.csv). O [registro da base](../resultados/experimentos/24/auditoria_base_24.json) contém seu ID e os motivos de exclusão.

Fonte original preparada: `datasets/homeobjects_sunrgbd_validado.zip`, sob a raiz Drive `TCC_Visao_Computacional`. SHA-256 `ada7fbee37b83f673b1b76bc57643d1fe4648941e0e8a7b895282216aed47676`. Conservar também a verificação de preparação interna. O [script de empacotamento](../scripts/preparar_home24.py) pode criar uma versão portátil do recorte Home, sem redistribuir o restante SUN contido no ZIP.

## SUN RGB-D

Fonte pública: [SUN RGB-D](https://rgbd.cs.princeton.edu/). A base foi preparada no 06 e usada nos ajustes 07/07b. Preservar a preparação e seus registros, além do ZIP Home/SUN validado usado posteriormente. As contagens e configurações das rodadas estão nas tabelas iniciais e nos respectivos notebooks.

SUN RGB-D incorpora imagens de outras fontes RGB-D. A relação com NYU precisa ser considerada quando se discute independência de modelos e testes. A lista de dados de treino de cada modelo deve acompanhar a interpretação das comparações.

## Open Images V7

Fonte pública: [downloads oficiais](https://storage.googleapis.com/openimages/web/download_v7.html). Os notebooks 08/08b/08c prepararam 100 mil imagens em lotes; o fluxo 10 registrou divisão de 80 mil/20 mil, com seed 42 e estratificação pela classe primária. Os IDs e a divisão estão em [manifestos](../datasets/manifestos/), preservados em GZIP sem alteração de conteúdo.

A raiz histórica é `datasets/openimages_v7_tcc_100k_v1/`. Preservar seus lotes, o manifesto de pacotes, a configuração de preparação, os metadados por imagem e a cobertura por classe. O [inventário](../datasets/arquivos_esperados.json) inclui os 100 lotes lidos pelo 21, com os hashes arquivados. Os recortes dos controles não equivalem a treinar todos os modelos nas 100 mil imagens.

O conjunto externo criado no 11 tem 1.000 imagens de validação e 300 de reserva. Fonte preparada:

`datasets/validacao_independente_11_v1/695b445e15c905642e2b276801b71132ced28e5eaee92d9fbcdd608f582f15a0/validacao_independente_11_695b445e15c9.zip`

SHA-256 `5af596546e2ad734c7b663f4567de2087d7fb42b41527d4852affdb4e7d2b96c`. A reserva de 300 foi aberta no 20. A validação de 1.000 foi utilizada em decisões dos controles anteriores e não deve ser descrita como teste intocado depois disso. [Manifesto original](../resultados/experimentos/11/manifesto_validacao_nova.json).

## NYU Depth V2, ramo 22/23

Fonte pública: [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html). A preparação extraiu imagens RGB e caixas das máscaras visíveis para seis classes. O ramo usa 634 imagens de treino, 161 de VAL e 654 de TEST, com partição de desenvolvimento por cenas e sem sobreposição de cenas registrada nessa separação.

| Arquivo preparado | Bytes | SHA-256 |
| --- | ---: | --- |
| `desenvolvimento_nyu22.zip` | 295.551.712 | `0e677c1e49fae0d48c9de4922ad63a84087f5f21c24592f4e50f64b56a1089ae` |
| `reserva_nyu22.zip` | 244.091.212 | `64a25836d6665e9c80141ac9ae74c2c2b0fa455eb2289fe3a5fdf194ce317074` |

Pasta histórica sob a raiz Drive: `ramo_nyu22_23_v1/preparado/3aae8a862f234d1995c65b46/`. A preparação também pode estar em `nyu22_local`, conforme a execução local anterior. Preservar `atual22.json` e `base_nyu22.json` junto à preparação; o segundo está [arquivado no pacote 23](../resultados/experimentos/23/base_nyu22.json).

As fontes MAT, seus hashes e a referência do split oficial estão em `base_nyu22.json`. Os notebooks 22 Windows/Colab explicam como regenerar a preparação. O arquivo MAT original tem cerca de 3 GB; as imagens preparadas de desenvolvimento e reserva são menores.

## Entrega da base e política de versão

O repositório inclui os manifestos e as listas de divisão disponíveis. Os quatro grupos de pacotes de imagens foram localizados no Drive em 19/10/2026: Home/SUN, NYU preparada, Open Images 100 mil e Open Images validação/reserva. Os links e instruções estão em [ACESSO_A_BASE.txt](../ACESSO_A_BASE.txt).

O acesso no Drive é de **Leitor para contas do Centro Universitário Impacta que tenham o link**, com login institucional autorizado. A opção de compartilhamento para qualquer pessoa com o link não foi oferecida pela conta proprietária. Os hashes registrados devem ser conferidos após o download.

Ao recuperar os ZIPs, executar [verificar_pacotes.py](../scripts/verificar_pacotes.py). Para Home 24, executar também o preparo portátil e conservar seu relatório. Não criar uma divisão nova para acompanhar um resultado histórico. Se qualquer arquivo de imagem ou rótulo divergir do manifesto, registrar como uma versão de dados diferente antes de repetir um experimento.

Os termos da fonte se aplicam ao compartilhamento. A atribuição e a informação de licença encontrada para cada base estão em [avisos de terceiros](../AVISOS_DE_TERCEIROS.md).
