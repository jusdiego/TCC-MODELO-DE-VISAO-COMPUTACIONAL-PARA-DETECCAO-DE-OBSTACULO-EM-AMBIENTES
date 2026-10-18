def _rf24_predict_corrigido(model, rows, batch_size=4):
    n = len(core.NAMES)
    if n != 12 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Esta correção exige as 12 classes do notebook 24 e batch positivo.')
    contexto = getattr(model, 'model', None)
    configuracoes = [getattr(model, 'model_config', None), getattr(contexto, 'args', None)]
    contagens = [getattr(cfg, 'num_classes', None) for cfg in configuracoes]
    contagens = [v for v in contagens if v is not None]
    if not contagens or any(v != n for v in contagens):
        raise ValueError('O RF-DETR não declarou exatamente 12 classes: ' + str(contagens))
    nomes = getattr(contexto, 'class_names', None)
    if isinstance(nomes, dict):
        if set(nomes) != set(range(n)):raise ValueError('Índices dos nomes do RF-DETR incompatíveis.')
        nomes = [nomes[i] for i in range(n)]
    if nomes is None or list(nomes) != core.NAMES:
        raise ValueError('A ordem das 12 classes do RF-DETR não corresponde ao projeto.')

    cache = {}
    for start in range(0, len(rows), batch_size):
        bloco = rows[start:start + batch_size]
        saidas = model.predict([str(r['image']) for r in bloco], threshold=.01)
        if not isinstance(saidas, list):saidas = [saidas]
        if len(saidas) != len(bloco):raise ValueError('RF-DETR retornou quantidade incorreta de imagens.')
        for r, out in zip(bloco, saidas):
            ids = np.asarray(out.class_id)
            caixas = np.asarray(out.xyxy)
            confiancas = np.asarray(out.confidence)
            if ids.ndim != 1 or ids.dtype.kind not in 'iu' or caixas.shape != (len(ids), 4) or confiancas.shape != (len(ids),):
                raise ValueError('Formato das previsões RF-DETR incompatível em ' + r['id'])
            invalidos = ids[(ids < 0) | (ids > n)]
            if len(invalidos):
                raise ValueError('RF-DETR retornou IDs inesperados: ' + str(sorted(set(map(int, invalidos))))
                                 + '. A correção não descarta classes desconhecidas.')
            fundo = ids == n
            if fundo.any():
                nomes_saida = getattr(out, 'data', {}).get('class_name')
                if nomes_saida is None or np.asarray(nomes_saida).shape != ids.shape or not np.all(np.asarray(nomes_saida)[fundo] == '__background__'):
                    raise ValueError('O ID 12 não foi confirmado como __background__ pelo RF-DETR em ' + r['id'])
            valores = [
                {'classe_id': int(cid), 'classe': core.NAMES[int(cid)],
                 'bbox': [float(x) for x in caixa], 'confianca': float(conf)}
                for cid, caixa, conf in zip(ids[~fundo], caixas[~fundo], confiancas[~fundo])
            ]
            fluxo.ev.validate_pred(valores, r)
            cache[r['id']] = valores
            _RF24_AUDITORIA['inferencias_nesta_aplicacao'].append({
                'id': r['id'], 'role': r['role'], 'previsoes_nativas': len(ids),
                'background_removido': int(fundo.sum()), 'previsoes_12_classes': len(valores)})
        core.save(JOB['output'] / 'correcao_background_rf24.json', _RF24_AUDITORIA)
    return cache
