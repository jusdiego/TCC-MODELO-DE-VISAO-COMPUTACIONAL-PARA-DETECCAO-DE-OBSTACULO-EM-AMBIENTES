"""Relatório de seleção e verificação, sem promoção automática de modelo."""
from __future__ import annotations
from collections import Counter
import base64
import html
import json
from pathlib import Path

import diag11_core as core
import lim17_protocol as protocol

def grid_rows(grid):
    rows = []
    for r in grid:
        h, e = r['home']['alertas_por_objeto'], r['external']['alertas_por_objeto']
        p = protocol.validate_profile(r['thresholds'])
        rows.append({'id': r['id'], 'partition': 'calibracao', 'limiar_cadeira': p[2], 'limiar_mesa': p[3],
                     'home_precision': h['precision'], 'home_recall': h['recall'], 'home_f1': h['f1'],
                     'home_tp': h['tp'], 'home_fn': h['fn'], 'home_fp': h['fp'],
                     'home_alta_sem_alerta': r['home']['alta_sem_alerta'],
                     'externa_precision': e['precision'], 'externa_recall': e['recall'], 'externa_f1': e['f1'],
                     'externa_tp': e['tp'], 'externa_fn': e['fn'], 'externa_fp': e['fp'],
                     'externa_alta_sem_alerta': r['external']['alta_sem_alerta'], **r['checks']})
    return rows

def _table(rows, columns):
    if not rows:
        return '<p>Não há configuração selecionada para esta etapa.</p>'
    def cell(value):
        if isinstance(value, float):
            value = f'{value:.5f}'
        return html.escape(str(value))
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(label) + '</th>' for _, label in columns) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + cell(row.get(key, '')) + '</td>' for key, _ in columns) + '</tr>' for row in rows) + '</tbody></table>'

def plot_calibration(grid, rule, selection, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    selected = selection['selected']
    for ax, key, label, rule_key in zip(axes, ['home', 'external'], ['Home', 'Open Images externa'], ['home', 'externa']):
        for r in grid:
            a = r[key]['alertas_por_objeto']
            ax.scatter(100 * a['recall'], 100 * a['precision'], s=28,
                       color='#176b54' if r['checks']['elegivel'] else '#adb5bd', alpha=.8)
        if selected:
            a = selected[key]['alertas_por_objeto']
            ax.scatter(100 * a['recall'], 100 * a['precision'], marker='*', s=180, color='#b65e00', label='Escolha congelada')
            ax.legend(loc='best')
        limits = rule[rule_key]
        ax.axvline(100 * limits['recall_minimo'], color='#983b35', linestyle='--', linewidth=1)
        ax.axhline(100 * limits.get('precision_maior_que', limits.get('precision_minima')), color='#983b35', linestyle='--', linewidth=1)
        ax.set(xlabel='Recall de alertas (%)', ylabel='Precisão de alertas (%)', title=label + ' — calibração')
        ax.grid(alpha=.18)
    fig.suptitle('49 configurações; as linhas mostram requisitos locais de precisão e recall')
    target = Path(output) / 'grade_calibracao.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target

def write_report(output, summary, grid, rules, selection, comparisons):
    output = Path(output)
    core.save_json(output / 'resumo_calibracao_17.json', summary)
    core.save_csv(output / 'grade_calibracao.csv', grid_rows(grid))
    core.save_csv(output / 'comparacao_referencias_e_configuracao.csv', comparisons)
    failures = Counter(key for row in grid for key, value in row['checks'].items()
                       if key not in {'elegivel', 'metricas_com_suporte'} and value is False)
    core.save_json(output / 'criterios_nao_atendidos_na_grade.json', dict(failures))
    picture = plot_calibration(grid, rules['calibracao'], selection, output)
    image = base64.b64encode(picture.read_bytes()).decode()
    selected = selection['selected']
    threshold_text = 'Nenhuma configuração elegível na calibração.'
    if selected:
        p = protocol.validate_profile(selected['thresholds'])
        threshold_text = f'Candidato 10: cadeira {p[2]:.2f}; mesa {p[3]:.2f}; demais classes 0,25.'
    text = '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Resultado 17</title><style>'
    text += 'body{font:16px system-ui,sans-serif;color:#202a30;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.55}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:8px;border-bottom:1px solid #d9e0e3;text-align:left}th{background:#edf3f2}img{width:100%;height:auto}code{background:#edf2f4;padding:2px 5px}.status{background:#edf3f2;padding:16px;border-left:4px solid #176b54}</style>'
    text += '<h1>Experimento 17 — limiares por classe</h1><p class="status">' + html.escape(summary['mensagem']) + '</p>'
    text += '<p>' + html.escape(threshold_text) + '</p><p>Não houve treino, alteração de rótulos ou substituição automática de modelo.</p>'
    text += '<h2>Seleção e verificação</h2><p>A grade foi avaliada somente em calibração. Uma escolha, quando existente, foi congelada antes da verificação interna. Falhar não autoriza tentar a segunda colocada. As duas validações já participaram do desenvolvimento; a verificação não é teste final independente.</p>'
    text += '<p>Configurações elegíveis em calibração: ' + str(selection['eligible_count']) + ' de 49.</p>'
    text += '<img alt="Precisão e recall das 49 configurações somente na calibração" src="data:image/png;base64,' + image + '">'
    text += '<h2>Referências e configuração congelada</h2><p>Precisão e recall abaixo são proporções. O conjunto completo é descritivo e não foi usado para trocar a escolha.</p>'
    text += _table(comparisons, [('partition','Partição'), ('modelo','Modelo'), ('coorte','Coorte'),
                              ('precision','Precisão'), ('recall','Recall'), ('f1','F1'), ('alta_sem_alerta','Altas omitidas')])
    text += '<h2>Critérios da escolha</h2>' + '<pre>' + html.escape(json.dumps(summary.get('avaliacao_escolha'), ensure_ascii=False, indent=2)) + '</pre>'
    text += '<h2>Limites</h2><ul>' + ''.join('<li>' + html.escape(s) + '</li>' for s in summary['limites']) + '</ul>'
    text += '<p>O pacote contém os registros GT de validação, previsões compactas suficientes para esta grade, código e diários de decisão. Os caches completos e pesos ficam separados no Drive.</p>'
    text += '<p>Envie <code>pacote_calibracao_17.zip</code>. Não há formulário a preencher.</p></html>'
    (output / 'relatorio_calibracao_17.html').write_text(text, encoding='utf-8')
    (output / 'LEIA_RESULTADO.md').write_text('# Experimento 17\n\n' + summary['mensagem'] + '\n\n' + threshold_text + '\n\nEnvie pacote_calibracao_17.zip. A verificação é interna e já houve exposição prévia às validações. Nenhum modelo foi substituído.\n', encoding='utf-8')

def comparison_row(partition, model, cohort, summary):
    return {'partition': partition, 'modelo': model, 'coorte': cohort,
            **summary['alertas_por_objeto'], 'alta_ref': summary['alta_ref'],
            'alta_sem_alerta': summary['alta_sem_alerta']}
