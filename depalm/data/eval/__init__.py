# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import evaluate

from .eval_base import SimpleEvaluateMetricWrapper, MetricEvaluator, AacMetrics, AacMetricMeteor
from .vqa_accuracy import VQAAccuracyMetric
from .gqa_accuracy import GQAAccuracyMetric
from .cccwrapper import CiderMetric, CocoLangEval, SpiceMetric, MeteorMetric

# Each entry is a function () => metric object
METRICS = {
    'vqa_accuracy': lambda : VQAAccuracyMetric(),
    'gqa_accuracy': lambda : GQAAccuracyMetric(),
    # 'ROUGE': lambda : SimpleEvaluateMetricWrapper('rouge', ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']),
    'BLEU': lambda : SimpleEvaluateMetricWrapper('bleu', ['bleu'], ['brevity_penalty', 'length_ratio']),
    'G_BLEU': lambda : SimpleEvaluateMetricWrapper('google_bleu', {'google_bleu': 'g_bleu'}),
    'G_BLEU@1': lambda : SimpleEvaluateMetricWrapper('google_bleu', {'google_bleu': 'G_BLEU@1'}, metric_kwargs={'min_len': 1, 'max_len': 1}),
    'G_BLEU@2': lambda : SimpleEvaluateMetricWrapper('google_bleu', {'google_bleu': 'G_BLEU@2'}, metric_kwargs={'min_len': 1, 'max_len': 2}),
    'G_BLEU@3': lambda : SimpleEvaluateMetricWrapper('google_bleu', {'google_bleu': 'G_BLEU@3'}, metric_kwargs={'min_len': 1, 'max_len': 3}),
    'G_BLEU@4': lambda : SimpleEvaluateMetricWrapper('google_bleu', {'google_bleu': 'G_BLEU@4'}, metric_kwargs={'min_len': 1, 'max_len': 4}),
    'B@1': lambda : SimpleEvaluateMetricWrapper('bleu', {'bleu': 'B@1'}, metric_kwargs={'max_order': 1}),
    'B@2': lambda : SimpleEvaluateMetricWrapper('bleu', {'bleu': 'B@2'}, metric_kwargs={'max_order': 2}),
    'B@3': lambda : SimpleEvaluateMetricWrapper('bleu', {'bleu': 'B@3'}, metric_kwargs={'max_order': 3}),
    'B@4': lambda : SimpleEvaluateMetricWrapper('bleu', {'bleu': 'B@4'}, metric_kwargs={'max_order': 4}),
    'METEOR-1.0': lambda : SimpleEvaluateMetricWrapper('meteor', ['meteor']),
    'METEOR': lambda : MeteorMetric(), # METEOR-1.5
    'CIDEr': lambda : CiderMetric(),
    'SPICE': lambda : SpiceMetric(),
    'coco_eval': lambda : CocoLangEval(), # You need an install of 'language_evaluation' for this one (https://github.com/bckim92/language-evaluation)
    'acc_eval': lambda : AacMetrics(),
    'acc_meteor': lambda : AacMetricMeteor(),
}