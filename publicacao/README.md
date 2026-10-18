# Distribuição de arquivos grandes

O conteúdo extraído deste repositório é destinado ao Git. Os arquivos desta tabela são downloads de Release ou dados obtidos pela distribuição da fonte.

| Arquivo | Finalidade | Estado na preparação |
| --- | --- | --- |
| `TCC_Resultados_Para_Apresentacao.zip` | Evidências completas recuperadas: relatórios, galerias, tabelas, previsões e o peso final. | Já entregue separadamente; 348.920.870 bytes. |
| `modelo_rfdetr24.pth` | Download direto do peso RF-DETR Small. | Presente dentro do ZIP anterior; 127.488.377 bytes. |
| `base_home24_reproducao.zip` | Imagens/rótulos do recorte Home 24 em uma embalagem portátil. | Gerado pelo script quando o ZIP original for fornecido; não incluído nesta entrega. |
| `base_home24_reproducao.verificacao.json` | Hash, contagens e verificação do pacote portátil. | Produzido junto ao pacote, sem preencher planilhas. |
| `desenvolvimento_nyu22.zip` e `reserva_nyu22.zip` | Imagens/rótulos da preparação NYU utilizada. | Referências e hashes preservados; imagens não localizadas nesta entrega. |
| Lotes e metadados Open Images | Reproduzir os recortes e controles históricos. | Inventário e divisões incluídos; imagens não localizadas nesta entrega. |

[arquivos_release.json](arquivos_release.json) descreve os assets disponíveis. O inventário de bases fica em [datasets/arquivos_esperados.json](../datasets/arquivos_esperados.json). A pasta local `publicacao/assets/` é ignorada pelo Git e pode receber downloads e pacotes gerados.

O pacote completo de resultados é uma coleção de evidências recuperadas, não uma cópia integral de todas as bases e checkpoints de treinamento. Seus manifestos históricos podem listar arquivos ausentes da seleção de apresentação. Para a íntegra de uma rodada, conservar o pacote original correspondente.

Procedimento: [publicação no GitHub](../docs/06_publicacao_github.md). O texto de [NOTAS_RELEASE.md](NOTAS_RELEASE.md) pode ser usado como descrição da entrega; o estado da disponibilidade dos dados deve ser atualizado quando os uploads ocorrerem.
