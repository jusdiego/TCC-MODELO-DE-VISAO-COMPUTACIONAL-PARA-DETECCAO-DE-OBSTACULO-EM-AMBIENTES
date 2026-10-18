# Publicação no GitHub

Nome sugerido: `tcc-visao-computacional-acessibilidade`.

O pacote já contém a estrutura dos arquivos, documentação, índice, resultados, manifestos e regras de exclusão. Ele ainda não está vinculado a uma conta ou repositório remoto. A publicação começa pela pasta extraída que contém `README.md`.

## Publicar a pasta pelo GitHub Desktop

1. Extrair o ZIP da entrega para uma pasta de trabalho no Windows.
2. Abrir o GitHub Desktop e entrar na conta que será proprietária do repositório.
3. Adicionar a pasta extraída como repositório local. Se o programa informar que ela ainda não é um repositório Git, usar a opção para criá-lo nessa pasta existente, preservando seus arquivos.
4. Conferir que o README está na raiz e fazer o primeiro commit, por exemplo `Organiza notebooks, dados documentados e resultados do TCC`.
5. Publicar o repositório. A visibilidade pública permite consulta pela banca e por terceiros; a visibilidade privada permite compartilhar acesso com pessoas específicas.

Não adicionar outra pasta de imagens ou checkpoints antes da publicação inicial. Os arquivos grandes são distribuídos pelo Release descrito abaixo. O `.gitignore` já cobre ZIPs, pesos, bases brutas e configurações locais de autenticação.

## Alternativa por terminal

Com Git e GitHub CLI instalados e a conta autenticada, executar dentro da pasta extraída:

```powershell
git init -b main
git add .
git commit -m "Organiza notebooks, dados documentados e resultados do TCC"
gh repo create tcc-visao-computacional-acessibilidade --private --source . --remote origin --push
```

O comando cria um repositório privado na conta autenticada. Para publicação pública desde o início, substituir somente `--private` por `--public`. Se já existir um repositório com esse nome, usar o repositório existente e seu endereço como remote; não repetir um comando de criação esperando atualizar o conteúdo.

O Git exige nome/e-mail de autoria do commit configurados na máquina. O GitHub Desktop pode orientar essa configuração; não há credencial ou identidade de conta embutida nesta entrega.

## Publicar os arquivos grandes em um Release

O GitHub bloqueia arquivos acima de 100 MiB no Git comum. Upload pela interface web tem limite menor por arquivo. O peso final e os pacotes completos devem ser anexados como assets do Release, mantendo o código extraído no repositório.

1. No repositório publicado, abrir **Releases** e criar um Release com tag `v1.0.0` apontando para a versão documentada.
2. Usar as notas preparadas em [NOTAS_RELEASE.md](../publicacao/NOTAS_RELEASE.md).
3. Anexar `TCC_Resultados_Para_Apresentacao.zip`; opcionalmente, anexar também `modelo_rfdetr24.pth` como download direto.
4. Ao recuperar a base Home original, gerar o pacote portátil com `scripts/preparar_home24.py` e anexar `base_home24_reproducao.zip` e seu relatório, preservando os termos da fonte.
5. Para as demais bases, disponibilizar os arquivos preparados quando houver condições de redistribuição; em outros casos, oferecer o acesso pela fonte e as instruções de preparação já documentadas.

Os [metadados dos assets disponíveis](../publicacao/arquivos_release.json) contêm nomes, tamanhos e hashes. Os ZIPs de imagens ainda não estão disponíveis nesta entrega. Não anunciar uma base completa como publicada antes de anexá-la ou disponibilizar seu link de acesso.

O GitHub informa limite de 2 GiB por asset de Release. Para bases maiores, preservar lotes individuais ou usar a distribuição externa documentada. Não é necessário incluir o dataset Open Images inteiro como um único arquivo.

## Conferência após publicar

Abrir a página principal, conferir os links do índice, a exibição dos notebooks e o acesso aos assets. Acrescentar no README os endereços reais do repositório/Release quando existirem. Os números Home e NYU, a condição histórica do TEST e as limitações já estão documentados e devem acompanhar a apresentação.

Fontes oficiais, consultadas em 18/10/2026:

- [Arquivos grandes no GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
- [Releases e limites dos assets](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
- [GitHub CLI: criar repositório](https://cli.github.com/manual/gh_repo_create).
- [GitHub CLI: criar Release](https://cli.github.com/manual/gh_release_create).
