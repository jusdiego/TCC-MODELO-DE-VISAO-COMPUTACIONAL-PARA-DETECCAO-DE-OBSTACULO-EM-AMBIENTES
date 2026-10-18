# Limitações e alcance dos resultados

## Home histórico

O TEST Home de 403 imagens já havia sido consultado em fases anteriores, incluindo 03/05d e o fluxo 10. No 24, as decisões da rodada foram congeladas antes de abri-lo novamente. Essa ordem evita seleção com o TEST nessa rodada, mas não o transforma em reserva nova e independente.

Foram registrados três candidatos a semelhança perceptual entre treino e TEST, sem coincidência exata de pixels nesses casos. A auditoria não confirmou agrupamentos completos de cenas relacionadas. Os intervalos por imagem podem ser estreitos demais se houver correlação entre imagens.

A meta foi atingida pela estimativa central micro. O intervalo de recall inclui valores inferiores a 70%. Isso não garante que cada classe alcance a meta, nem que o desempenho seja mantido em qualquer residência ou câmera.

## Mudança de domínio e anotações

No NYU externo, o RF-DETR final apresentou P 70,82% e R 45,39% nas seis classes comuns. Esse resultado é diagnóstico de transferência, com classes e caixas visíveis diferentes da composição Home. O ramo 23 também não atingiu a meta própria.

SUN RGB-D inclui imagens de fontes como NYU. Uma comparação entre modelos treinados em fontes que podem compartilhar imagens não deve ser chamada de benchmark independente garantido. O 24 foi treinado apenas em Home, mas essa restrição não resolve retrospectivamente a proveniência de todos os comparadores.

Open Images tem cobertura por classe e marcações específicas, inclusive caixas de grupos. Os protocolos do projeto precisam ser lidos com suas regras de cobertura. Não está demonstrado nesta organização que toda a avaliação histórica implementou a regra oficial de `IsGroupOf`; não atribuir uma alteração de resultado a esse ponto sem uma verificação concreta.

As auditorias automáticas sinalizaram divergências, sem provar todos os erros de rótulo. O pacote 13 não documenta revisão humana concluída. A auditoria 15 não aprovou uma nova base reanotada.

## Comparação entre modelos

Candidato 10, 21/época 21 e RF-DETR 24 diferem em dados de treino, capacidade, resolução, otimização e pós-processamento. O teste comum permite comparar as configurações completas nesse conjunto; não identifica isoladamente a causa dos ganhos.

As métricas de alertas dos ciclos anteriores não são equivalentes às métricas micro do detector no 24. Também não são equivalentes a mAP50 ou mAP50–95. O limiar de confiança não foi demonstrado como probabilidade calibrada de acerto.

## Aplicação e execução

A política V2 estima prioridade por classe e geometria 2D. Não mede distância real, não reconhece todos os obstáculos possíveis e não comprova uma rota segura. Não há validação com usuários documentada nestes pacotes. O trabalho sustenta um protótipo experimental de percepção e mensagens, com limites de generalização registrados.

O pacote 21 preserva um estado pausado/retomável. Alguns notebooks não têm saídas gravadas no arquivo e devem ser associados aos pacotes de evidências. O notebook 24 Colab foi recomposto do código arquivado da execução, sem fabricar saídas. As imagens completas e alguns checkpoints históricos dependem dos arquivos originais conservados fora deste repositório.
