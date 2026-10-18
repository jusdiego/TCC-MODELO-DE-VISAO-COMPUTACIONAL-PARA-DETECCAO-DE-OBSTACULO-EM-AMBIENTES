# Distribuição de arquivos grandes

O código e a documentação estão no [repositório público](https://github.com/jusdiego/tcc-visao-computacional-acessibilidade). O pacote de resultados e o peso final foram publicados na [versão v1.0.0](https://github.com/jusdiego/tcc-visao-computacional-acessibilidade/releases/tag/v1.0.0); os anexos podem ser baixados sem login. A distribuição das bases completas de imagens será definida posteriormente.

| Arquivo | Finalidade | Disponibilidade |
| --- | --- | --- |
| [TCC_Resultados_Para_Apresentacao.zip](https://github.com/jusdiego/tcc-visao-computacional-acessibilidade/releases/download/v1.0.0/TCC_Resultados_Para_Apresentacao.zip) | Evidências completas recuperadas: relatórios, galerias, tabelas, previsões e o peso final. | Publicado na versão v1.0.0; 348.920.870 bytes. |
| [modelo_rfdetr24.pth](https://github.com/jusdiego/tcc-visao-computacional-acessibilidade/releases/download/v1.0.0/modelo_rfdetr24.pth) | Download direto do peso RF-DETR Small, também incluído no ZIP de resultados. | Publicado na versão v1.0.0; 127.488.377 bytes. |
| `base_home24_reproducao.zip` | Imagens/rótulos do recorte Home 24 em uma embalagem portátil. | Gerado pelo script quando o ZIP original for fornecido; não incluído nesta entrega. |
| `base_home24_reproducao.verificacao.json` | Hash, contagens e verificação do pacote portátil. | Produzido junto ao pacote, sem preencher planilhas. |
| `desenvolvimento_nyu22.zip` e `reserva_nyu22.zip` | Imagens/rótulos da preparação NYU utilizada. | Referências e hashes preservados; imagens não localizadas nesta entrega. |
| Lotes e metadados Open Images | Reproduzir os recortes e controles históricos. | Inventário e divisões incluídos; imagens não localizadas nesta entrega. |

[arquivos_release.json](arquivos_release.json) descreve os assets disponíveis. O inventário de bases fica em [datasets/arquivos_esperados.json](../datasets/arquivos_esperados.json). A pasta local `publicacao/assets/` é ignorada pelo Git e pode receber downloads e pacotes gerados.

O pacote completo de resultados é uma coleção de evidências recuperadas, não uma cópia integral de todas as bases e checkpoints de treinamento. Seus manifestos históricos podem listar arquivos ausentes da seleção de apresentação. Para a íntegra de uma rodada, conservar o pacote original correspondente.

Procedimento: [publicação no GitHub](../docs/06_publicacao_github.md). As [notas preparadas](NOTAS_RELEASE.md) descrevem a entrega; a descrição publicada fica na [Release v1.0.0](https://github.com/jusdiego/tcc-visao-computacional-acessibilidade/releases/tag/v1.0.0). A tag identifica o commit original da versão. Correções posteriores de documentação e integridade ficam na branch `main`, com registro no histórico de commits.
