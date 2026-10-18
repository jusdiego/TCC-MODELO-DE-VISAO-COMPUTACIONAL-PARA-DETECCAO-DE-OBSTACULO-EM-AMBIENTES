# Integridade e proveniência

## Materiais de referência

Esta organização parte dos arquivos `TCC_Notebooks_Organizados.zip` e `TCC_Resultados_Para_Apresentacao.zip`, que reuniram os notebooks e os pacotes históricos disponíveis. A preparação do repositório não executou novos treinamentos nem novas inferências.

O [catálogo de origens](../auditoria/catalogo_origens.csv) registra cada cópia, a origem e seu SHA-256. Os manifestos de dados em GZIP preservam exatamente o conteúdo descompactado, com hash correspondente no catálogo. O resumo final em `resultados/resumo_final.json` e as tabelas da documentação são derivados dos registros do 24, sem substituir as fontes originais.

`SHA256SUMS` permite conferir esta distribuição. `.gitattributes` desativa conversão automática de finais de linha, para preservar os bytes históricos em clones no Windows. Após editar arquivos intencionalmente, o hash da nova versão precisa ser atualizado; uma divergência não deve ser ignorada para afirmar identidade com a versão anterior.

## Organização dos notebooks

São 32 notebooks principais e 39 variantes. [indice_notebooks.csv](indice_notebooks.csv) informa ambiente, número de saídas gravadas, situação, caminho e origem. A organização anterior retirou instruções operacionais de compartilhamento em conversas e reescreveu textos para descrever os procedimentos de investigação. Parâmetros, lógica científica, métricas e limitações foram preservados.

Os registros dessa revisão permanecem em [organizacao_anterior](../auditoria/organizacao_anterior/). Os hashes dos notebooks editados diferem de seus arquivos anteriores. Em módulos embutidos, textos alterados podem mudar hashes de código e identificadores de uma nova rodada. Para a reprodução exata de uma execução histórica, os pacotes originais preservam o código de referência.

O notebook 24 Colab foi recomposto a partir do código arquivado da execução e da organização de células do notebook 24. O bundle original de nove módulos corresponde ao hash registrado `e69d0d10101fdf4ecb3adf50b25c6c1cbc9b07151b35d0746f459c5bfcf7e4fa`. A correção de background foi mantida. Não foram produzidas saídas artificiais para apresentar esse arquivo como o notebook original executado. A variante Kaggle corrigida também foi preservada.

Houve assistência de IA na elaboração de código e documentação ao longo do projeto. Os registros de execução e as verificações explicitadas sustentam as afirmações de resultados; não se presume revisão humana de rótulos onde ela não foi registrada.

## Evidências selecionadas e pacote completo

O Git contém os notebooks, código e registros tabulares principais. Os HTMLs, galerias, caches de previsões e arquivos auxiliares do conjunto recuperado permanecem no pacote completo destinado ao Release. A seleção não pretende representar cada peso antigo ou cada arquivo bruto como disponível no Git.

Manifestos internos dos pacotes antigos podem listar arquivos omitidos da cópia de apresentação. Para esta distribuição, usar seu próprio catálogo e `SHA256SUMS`; para a integridade integral de uma rodada, usar o pacote histórico original e seu manifesto. A identidade do peso final é preservada em [modelo_final.json](../modelos/modelo_final.json).

## Conferências desta entrega

O [registro de preparação](../auditoria/preparacao_repositorio.json) identifica o escopo. O [relatório de verificação](../auditoria/verificacao_repositorio.json) informa as conferências realizadas na montagem do pacote. Os scripts novos de empacotamento/conferência foram verificados com arquivos temporários sintéticos, separados das evidências científicas. Não foram usados dados sintéticos para calcular os resultados do TCC.

Os ZIPs completos de imagens não estavam entre os arquivos recuperados para esta organização. Sua ausência está registrada no inventário; não foram substituídos por imagens extraídas de galerias nem por uma divisão nova.
