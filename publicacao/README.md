# Distribuição de arquivos grandes

O código e a documentação estão no [repositório público](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES). O pacote de resultados e o peso final foram publicados na [versão v1.0.0](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES/releases/tag/v1.0.0); os anexos podem ser baixados sem login. As bases preparadas são distribuídas pelos links de [ACESSO_A_BASE.txt](../ACESSO_A_BASE.txt), em modo Leitor para contas do Centro Universitário Impacta com o link. Esses downloads exigem login institucional autorizado.

| Arquivo | Finalidade | Disponibilidade |
| --- | --- | --- |
| [TCC_Resultados_Para_Apresentacao.zip](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES/releases/download/v1.0.0/TCC_Resultados_Para_Apresentacao.zip) | Evidências completas recuperadas: relatórios, galerias, tabelas, previsões e o peso final. | Publicado na versão v1.0.0; 348.920.870 bytes. |
| [modelo_rfdetr24.pth](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES/releases/download/v1.0.0/modelo_rfdetr24.pth) | Download direto do peso RF-DETR Small, também incluído no ZIP de resultados. | Publicado na versão v1.0.0; 127.488.377 bytes. |
| `homeobjects_sunrgbd_validado.zip` | ZIP fonte Home/SUN exigido pelos notebooks históricos. | Disponível no Drive; acesso institucional de leitor, conforme o [TXT](../ACESSO_A_BASE.txt). |
| `base_home24_reproducao.zip` | Imagens/rótulos do recorte Home 24 em uma embalagem portátil. | Gerado pelo script quando o ZIP original for fornecido; não incluído nesta entrega. |
| `base_home24_reproducao.verificacao.json` | Hash, contagens e verificação do pacote portátil. | Produzido junto ao pacote, sem preencher planilhas. |
| `desenvolvimento_nyu22.zip` e `reserva_nyu22.zip` | Imagens/rótulos da preparação NYU utilizada. | Disponíveis no Drive; referências e hashes preservados. Acesso institucional de leitor. |
| Lotes e metadados Open Images | Reproduzir os recortes e controles históricos. | Disponíveis no Drive; 100 lotes, metadados e pacote de validação/reserva. Acesso institucional de leitor. |

[arquivos_release.json](arquivos_release.json) descreve os assets disponíveis. O inventário de bases fica em [datasets/arquivos_esperados.json](../datasets/arquivos_esperados.json). A pasta local `publicacao/assets/` é ignorada pelo Git e pode receber downloads e pacotes gerados.

O pacote completo de resultados é uma coleção de evidências recuperadas, não uma cópia integral de todas as bases e checkpoints de treinamento. Seus manifestos históricos podem listar arquivos ausentes da seleção de apresentação. Para a íntegra de uma rodada, conservar o pacote original correspondente.

Procedimento: [publicação no GitHub](../docs/06_publicacao_github.md). As [notas preparadas](NOTAS_RELEASE.md) descrevem a entrega; a descrição publicada fica na [Release v1.0.0](https://github.com/jusdiego/TCC-MODELO-DE-VISAO-COMPUTACIONAL-PARA-DETECCAO-DE-OBSTACULO-EM-AMBIENTES/releases/tag/v1.0.0). A tag identifica o commit original da versão. Correções posteriores de documentação e integridade ficam na branch `main`, com registro no histórico de commits.
