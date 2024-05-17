# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Dict, Any

from .eval_base import Metric
from .cccmetrics.cider_scorer import CiderScorer
from .cccmetrics.spice.spice import SpiceScorer
from .cccmetrics.meteor.meteor import MeteorScorer


class CiderMetric(Metric):
    def __init__(self, n_grams=4):
        self.n_grams = n_grams

    def _compute_raw_text(self, references: List[List[str]], predictions: List[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        cider_scorer = CiderScorer(n=self.n_grams)
        for ref, pred in zip(references, predictions):
            cider_scorer.cook_append(pred, ref)
        score, _ = cider_scorer.compute_score()
        return {'CIDEr': score}, {}


class SpiceMetric(Metric):
    def _compute_raw_text(self, references: List[List[str]], predictions: List[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        scorer = SpiceScorer()
        score, _ = scorer.compute_score(references, predictions)
        return {'SPICE': score}, {}


class MeteorMetric(Metric):
    def _compute_raw_text(self, references: List[List[str]], predictions: List[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        scorer = MeteorScorer()
        score, _ = scorer.compute_score(references, predictions)
        return {'METEOR': score}, {}


class CocoLangEval(Metric):
    def __init__(self):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator()

    def _compute_raw_text(self, references: List[List[str]], predictions: List[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        scores = self.evaluator.run_evaluation(predictions, references)
        scores = {'le_' + key: val for key, val in scores.items()}
        return scores, {}