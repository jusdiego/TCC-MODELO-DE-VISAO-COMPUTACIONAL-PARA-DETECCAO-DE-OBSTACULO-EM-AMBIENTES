# v1.0.0 — investigação e resultados do TCC

Organização da investigação de detecção de objetos domésticos e mensagens heurísticas de atenção para ambientes internos. Inclui 32 notebooks principais, 39 variantes, código arquivado, metodologia, decisões dos experimentos, resultados e manifestos/divisões das bases.

Resultado RF-DETR Small no Home TEST histórico, 403 imagens e 12 classes: precisão 81,84%, recall 71,40%, F1 76,26%, confiança mínima 0,50. A meta foi atingida na estimativa pontual; o intervalo de recall de 95% é 69,46%–73,50%. O conjunto Home é histórico e já havia sido consultado no projeto.

No diagnóstico NYU de 654 imagens e seis classes comuns, o mesmo peso/limiar teve precisão 70,82%, recall 45,39% e F1 55,33%. Esses resultados delimitam a generalização para outra base. As mensagens V2 são heurísticas 2D, sem validação de segurança de locomoção com usuários.

O arquivo `TCC_Resultados_Para_Apresentacao.zip` reúne as evidências recuperadas e o peso final em `modelo_final/modelo_rfdetr24.pth`. O repositório contém os manifestos das bases; os ZIPs completos de imagens não acompanham o pacote de código desta versão. A documentação descreve os arquivos necessários e a preparação.

Hashes dos arquivos distribuídos devem corresponder a `publicacao/arquivos_release.json`. Para conferir o código e as tabelas da versão, usar `SHA256SUMS` e `python scripts/verificar_repositorio.py`.
