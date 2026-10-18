# Publicação no GitHub

## Estado atual

O [repositório `jusdiego/tcc-visao-computacional-acessibilidade`](https://github.com/jusdiego/tcc-visao-computacional-acessibilidade) está publicado com visibilidade pública. A branch `main` contém os notebooks, a documentação, os resultados tabulares, os manifestos e as regras de exclusão.

A [Release v1.0.0](https://github.com/jusdiego/tcc-visao-computacional-acessibilidade/releases/tag/v1.0.0) disponibiliza `TCC_Resultados_Para_Apresentacao.zip` e `modelo_rfdetr24.pth`. Os links, tamanhos e hashes dos anexos estão em [arquivos_release.json](../publicacao/arquivos_release.json). Os anexos podem ser baixados sem login.

A tag `v1.0.0` identifica o commit original da versão. Correções posteriores de documentação e integridade são registradas na branch `main`. Uma futura versão deve usar uma nova tag.

## Atualizar pelo GitHub Desktop no Windows

1. Entrar na conta com acesso ao projeto. No primeiro acesso, usar **File → Clone repository → URL** com `https://github.com/jusdiego/tcc-visao-computacional-acessibilidade.git` e escolher uma pasta local.
2. Abrir o repositório no Desktop, selecionar a branch `main` e usar **Fetch origin** e **Pull origin**, quando disponível, antes de editar.
3. Usar **Repository → Show in Explorer** para abrir a cópia local e copiar os arquivos para seus caminhos correspondentes, mantendo `README.md` na raiz.
4. Após edições intencionais, atualizar as entradas correspondentes em `SHA256SUMS`. Preservar as cópias históricas e os hashes do catálogo de origens.
5. Na raiz, executar `python scripts/verificar_repositorio.py` e conferir o resultado.
6. Revisar os arquivos em **Changes**, escrever a mensagem e usar **Commit to main**.
7. Usar **Push origin** e conferir os arquivos no GitHub.

O `.gitignore` exclui ZIPs, pesos, bases brutas e configurações locais de autenticação. O `.gitattributes` preserva os bytes históricos também em clones no Windows. As bases completas de imagens continuam fora do Git; sua distribuição será definida posteriormente.

## Alternativa por terminal

Para obter uma cópia local com Git instalado e acesso autenticado:

```powershell
git clone https://github.com/jusdiego/tcc-visao-computacional-acessibilidade.git
cd tcc-visao-computacional-acessibilidade
```

Em uma cópia existente, sincronizar antes de editar:

```powershell
git switch main
git pull --ff-only origin main
```

Depois de copiar ou editar os arquivos e atualizar as entradas de integridade correspondentes:

```powershell
python scripts/verificar_repositorio.py
git add .
git commit -m "Atualiza documentação e arquivos do TCC"
git push origin main
```

## Arquivos grandes e futuras versões

O peso final e o pacote de apresentação já estão nos [anexos da Release v1.0.0](https://github.com/jusdiego/tcc-visao-computacional-acessibilidade/releases/tag/v1.0.0). Os arquivos grandes devem ser anexados a Releases; o Git comum bloqueia arquivos acima de 100 MiB.

Para uma futura versão:

1. Conferir os arquivos e enviar o commit para `main`.
2. Em **Releases**, criar uma nova tag apontando para o commit que será distribuído.
3. Preparar as notas a partir de [NOTAS_RELEASE.md](../publicacao/NOTAS_RELEASE.md), ajustando a versão e descrevendo as alterações.
4. Anexar os pacotes e pesos correspondentes, aguardar o upload e conferir tamanhos e hashes antes de publicar.
5. Atualizar os links e metadados da distribuição no repositório.

Os ZIPs completos de imagens ainda não foram publicados. Ao recuperar a base Home original, o script `scripts/preparar_home24.py` pode gerar `base_home24_reproducao.zip` e seu relatório. As demais bases mantêm suas instruções e condições de distribuição, conforme [bases de dados](../datasets/README.md) e [avisos de terceiros](../AVISOS_DE_TERCEIROS.md).

## Conferência

Na branch `main`, conferir os links do índice, os notebooks e `python scripts/verificar_repositorio.py`. Na Release, conferir os anexos e os hashes registrados. A condição histórica do TEST Home, os resultados externos NYU e as limitações documentadas devem acompanhar a apresentação.

Fontes oficiais:

- [Arquivos grandes no GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
- [Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
- [Clonar e atualizar com GitHub Desktop](https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop).
