# Bases de dados e divisões dos experimentos

Para reproduzir uma avaliação, são necessários imagens, rótulos, taxonomia e a divisão usada naquela rodada. Baixar a versão atual de uma base e gerar outra divisão produz um experimento diferente.

## O que acompanha o repositório

- Manifesto Home do experimento 24, sem alteração de conteúdo, comprimido em [home24.json.gz](manifestos/home24.json.gz).
- Listas de treino/VAL/TEST Home com caminho original e hashes de imagem/rótulo em [divisoes/](divisoes/).
- Manifesto de desenvolvimento NYU, listas por partição e IDs de TEST em [manifestos/](manifestos/) e [divisoes/](divisoes/).
- Divisão Open Images histórica de 100 mil IDs, em JSON e CSV comprimidos. A cobertura/anotação original continua documentada nos resultados e nos notebooks de preparação.
- [Taxonomia Home](classes_home.json), [mapeamento NYU → Home](mapeamento_nyu_home.csv) e [inventário de arquivos esperados](arquivos_esperados.csv).

**Os links para baixar as bases estão em [ACESSO_A_BASE.txt](../ACESSO_A_BASE.txt).** Em 19/10/2026, foram localizados no Drive o ZIP Home/SUN, os dois ZIPs NYU, os 100 lotes Open Images e o ZIP de validação/reserva. O acesso por link está configurado como **Leitor para contas do Centro Universitário Impacta** e exige login institucional autorizado; a conta proprietária não ofereceu acesso por link para qualquer pessoa.

As imagens completas ficam no Drive; os manifestos e divisões acompanham o Git. Os pacotes de resultados contêm exemplos/galerias e previsões e não substituem as bases completas. [Bases e preparação](../docs/07_bases.md) detalha as versões e os arquivos necessários.

## Base principal final: Home 24

| Partição | Imagens | Identificação |
| --- | ---: | --- |
| Treino | 1.818 | `divisoes/home24_train.csv` |
| Validação | 456 | `divisoes/home24_val.csv` |
| TEST histórico | 403 | `divisoes/home24_test.csv` |

O recorte usa 12 classes. O manifesto registra seis exclusões do treino original de 1.824 imagens. O TEST permanece histórico e previamente consultado, inclusive após seu empacotamento para reprodução.

O ZIP fonte exigido pelos notebooks históricos chama-se `homeobjects_sunrgbd_validado.zip`, na pasta Drive `TCC_Visao_Computacional/datasets/`, SHA-256:

`ada7fbee37b83f673b1b76bc57643d1fe4648941e0e8a7b895282216aed47676`

Esse arquivo também pode conter SUN RGB-D. O preparo do 24 seleciona somente o Home definido no manifesto.

Para gerar um pacote portátil com **as mesmas imagens e os mesmos rótulos** do Home 24, na raiz do repositório:

```powershell
python scripts/preparar_home24.py --origem "C:\caminho\homeobjects_sunrgbd_validado.zip"
```

O script confere o hash do ZIP fonte e o conteúdo de cada imagem/rótulo, usa a divisão existente e grava `publicacao/assets/base_home24_reproducao.zip`. Ele não treina, não refaz a seleção e não exige preencher anotações. O caminho do arquivo é a única entrada necessária. Use `--somente-conferir` para verificar sem gerar o pacote.

O pacote portátil não substitui o ZIP fonte nos notebooks históricos, que conferem seu hash original. Ele serve para distribuir a versão usada e para uma reprodução com caminhos adaptados explicitamente. Consulte o relatório de geração incluído no pacote.

## Conferência dos arquivos originais

Para procurar e conferir os ZIPs esperados dentro da pasta que contém os dados do projeto:

```powershell
python scripts/verificar_pacotes.py --origem "C:\caminho\TCC_Visao_Computacional"
```

O relatório distingue arquivos presentes e compatíveis, ausentes e divergentes. A conferência não baixa dados nem altera os originais.

## Distribuição

Os manifestos pequenos ficam no Git. Pacotes de imagens podem ser anexados a um Release ou oferecidos por um link de download, preservando a versão, os hashes e os termos da origem. Não foi atribuída uma licença única às imagens de Home, SUN RGB-D, Open Images e NYU. As fontes e atribuições estão em [AVISOS_DE_TERCEIROS.md](../AVISOS_DE_TERCEIROS.md).
