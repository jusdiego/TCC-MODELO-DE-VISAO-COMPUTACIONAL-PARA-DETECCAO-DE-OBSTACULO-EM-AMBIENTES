# Peso final

O arquivo final é `modelo_rfdetr24.pth`, RF-DETR Small, 12 classes Home, resolução 512, melhor época 13 e confiança mínima escolhida de 0,50. A identidade completa está em [modelo_final.json](modelo_final.json).

O peso tem 127.488.377 bytes e SHA-256 `d481dcf7e96479b338cb42fd9199d8a4cccde8fbc8ed8e341f528c1e13d51900`. Deve ser distribuído em um Release, ou obtido de `TCC_Resultados_Para_Apresentacao.zip/modelo_final/modelo_rfdetr24.pth`.

Guardar o download em `modelos/pesos/`, pasta ignorada pelo Git. O peso `.pth` deve ser carregado com RF-DETR, conforme o notebook 24 e seu ambiente registrado. Não é um peso YOLO. A avaliação histórica aplicou a correção de background arquivada no experimento.

O pacote de apresentação não inclui os pesos de todos os modelos antigos. A comparação completa de inferência do notebook 24 também requer candidato 10 e 21/época 21, cujos hashes e caminhos estão no protocolo e no código arquivado.
