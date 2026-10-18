"""Consolidação e protocolo da avaliação posterior, sem abrir a reserva."""
from __future__ import annotations
import hashlib,json

VERSION='19-consolidacao-demonstracao-1.0.0'
MODELS=('baseline','candidato_10')
COHORTS=('home_val_historica','openimages_val_nova')
SPEC={'version':VERSION,'models':list(MODELS),'confidence':.25,'policy':'v2','iou_match':.5,
      'training':False,'inference':False,'demo_images_per_cohort':12,'demo_per_category':2,
      'demo_seed':'19-galeria-42','demo_categories':['mais_tp_candidato10','mais_tp_baseline',
          'omissao_alta','falso_alerta','acerto_sem_erro','sem_alerta_referencia'],
      'selection':'categorias ilustrativas com ordem por SHA; sem trocar pesos/limiares',
      'evaluation':'duas validacoes ja expostas; todas as imagens, sem usar a galeria como metrica',
      'final_reserve_opened':False,'new_weight_selection':False}
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()

def final_protocol(expected):
    """Contagens absolutas das 1.000 VAL não viram limites artificiais nas 300."""
    return {'version':'20-plano-avaliacao-final-congelado-em-19-v1','status':'planejado; reserva nao consultada no 19',
        'primary_model':'baseline','comparison_model':'candidato_10',
        'weight_sha256':{'baseline':expected['baseline'],'candidato_10':expected['candidato']},
        'dataset_sha256':expected['dataset_novo'],'archive_sha256':expected['zip_novo'],
        'dataset_location':expected['pasta_validacao']+'/'+expected['arquivo_validacao'],
        'role':'reserva_futura','images':300,'population':'todos os 300 IDs do papel reserva_futura no manifesto autenticado; sem substituicoes',
        'classes':'mesmo mapeamento de 12 classes; cobertura parcial original preservada',
        'policy_sha256':expected['politica'],'matcher_sha256':expected['associador'],
        'inference':{'ultralytics':'8.4.150','conf':.001,'imgsz':640,'rect':True,'iou':.7,'max_det':300,'half':False,'batch':1},
        'alert_confidence':.25,'match_iou':.5,'policy':'v2',
        'primary_estimands':['recall de alertas por objeto do baseline','delta recall candidato10 menos baseline'],
        'secondary':['precisao','F1','omissoes altas: contagem e taxa por objetos GT de prioridade alta',
            'metricas por classe e suporte','contagens TP/FN/FP e denominadores'],
        'confidence_intervals':{'method':'bootstrap pareado de grupos de imagens; manter bytes iguais e pHash64 distancia<=4 no mesmo grupo por ligacao transitiva',
            'repetitions':2000,'seed':19002026,'confidence':.95,'interval':'percentil 2.5 e 97.5; grupos reamostrados com reposicao; mesmos grupos para ambos os modelos',
            'zero_denominator':'replica sem suporte fica indefinida; informar quantidade de replicas validas por metrica'},
        'low_support':'informar contagens; sinalizar classe com menos de 30 GT, sem ocultar resultados',
        'weight_or_threshold_tuning_after_reserve':False,'second_choice_after_failure':False,
        'deployment_decision':'nenhuma troca automatica; caracterizacao dos dois modelos predefinidos',
        'historical_rules':'os seis limites das validacoes ficam no desenvolvimento; nao aplicar limites absolutos de 1.000 imagens em 300',
        'home_test':'fora desta avaliacao primaria; historico original do baseline nao comprovado',
        'openimages_test20k':'fora desta etapa','training':False,
        'limits':['A independencia refere-se a reserva preservada no fluxo documentado; nao comprova ausencia de exposicao em todo pretreino original.',
                  'Nao valida distancia, risco fisico ou uso assistivo em ambiente real.']}
