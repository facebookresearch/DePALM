# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Tuple, Any, Union
import evaluate
from pathlib import Path
from aac_metrics import evaluate as aac_evaluate

from ..data.data_base import DataRow, DataRowMultipleOutputs
from .cccmetrics.tokenizer.ptbtokenizer import PTBTokenizer
from .cccmetrics.cccpaths import JAVA_PATH, CORENLPPATH


class Metric:
    def compute(self, references: List[DataRow], predictions: List[str], **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
        output_text_lists = [
            ref.output_text_list if isinstance(ref, DataRowMultipleOutputs) else ref.output_text
            for ref in references
        ]
        return self._compute_raw_text(output_text_lists, predictions, **kwargs)

    def _compute_raw_text(self, references: List[List[str]], predictions: List[str], **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
        raise NotImplementedError

class MetricEvaluator:
    def __init__(self, metrics: List[str], tokenize=False):
        from . import METRICS
        self.task_metrics = {
            metric_name: METRICS[metric_name]()
            for metric_name in metrics
        }

        self.tokenizer = None
        if tokenize:
            self.tokenizer = PTBTokenizer()

    def evaluate(self, references: List[DataRow], predictions: List[str]) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        # Tokenization
        if self.tokenizer:
            # Tokenize only output_text and output_text_list
            references = DataRow.stack_rows(references)
            if references.output_text[0] is not None:
                references.output_text = self.tokenizer(references.output_text)
            if isinstance(references, DataRowMultipleOutputs):
                references.output_text_list = self.tokenizer(references.output_text_list)
            references = references.unstack()

            predictions = self.tokenizer(predictions)

        metric_dict, info_dict = {}, {}
        for metric_name, metric in self.task_metrics.items():
            m_dict, i_dict = metric.compute(references, predictions)
            metric_dict.update(m_dict)
            info_dict[metric_name] = m_dict | i_dict
        return metric_dict, info_dict



##### Wrapper around 'evaluate' metrics #####

class BaseEvaluateMetricWrapper(Metric):
    def __init__(self, metric, default_value=0., metric_kwargs=None):
        if isinstance(metric, str):
            metric = evaluate.load(metric)
        self.metric = metric
        self.default_value = float(default_value)
        self.metric_kwargs = metric_kwargs or {}

    def compute(self, references: List[DataRow], predictions: List[str], **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if isinstance(references[0], DataRowMultipleOutputs):
            references = [ref.output_text_list for ref in references]
        else:
            references = [ref.output_text for ref in references]
        try:
            metrics_vals = self.metric.compute(references=references, predictions=predictions, **self.metric_kwargs, **kwargs)
        except ZeroDivisionError: # Hot fix for BLEU score of zero
            return self.extract_metrics(self.default_value)
        return self.extract_metrics(metrics_vals)

    def extract_metrics(self, metrics_vals: Union[Dict, float]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        raise NotImplementedError

class SimpleEvaluateMetricWrapper(BaseEvaluateMetricWrapper):
    def __init__(self, metric, metric_keys: Union[Dict[str, str], List[str]], info_keys: List[str]=None, **kwargs):
        super().__init__(metric, **kwargs)
        if isinstance(metric_keys, list):
            metric_keys = {key: key for key in metric_keys}
        self.metric_keys = metric_keys
        self.info_keys = info_keys or []

    def extract_metrics(self, metrics_vals: Union[Dict, float]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        metric_dict, info_dict = {}, {}
        if isinstance(metrics_vals, dict):
            for key, val in metrics_vals.items():
                if key in self.metric_keys:
                    metric_dict[self.metric_keys[key]] = val
                if key in self.info_keys or key in self.metric_keys:
                    info_dict[key] = val
        else:
            assert isinstance(metrics_vals, float)
            metric_dict = {key: metrics_vals for key in self.metric_keys}
        return metric_dict, info_dict


##### Imported metrics (from libraries) #####

class AacMetrics(Metric):
    def compute(self, references: List[DataRow], predictions: List[str], **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
        references = [
            ref.output_text_list if isinstance(ref, DataRowMultipleOutputs) else [ref.output_text]
            for ref in references
        ]
        scores, _ = aac_evaluate(predictions, references, java_path=JAVA_PATH, cache_path=CORENLPPATH)
        scores = {key: float(val) for key, val in scores.items()}
        scores['B@1'] = scores.pop('bleu_1')
        scores['B@2'] = scores.pop('bleu_2')
        scores['B@3'] = scores.pop('bleu_3')
        scores['B@4'] = scores.pop('bleu_4')
        scores['CIDEr'] = scores.pop('cider_d')
        scores['SPICE'] = scores.pop('spice')
        scores['SPIDEr'] = scores.pop('spider')
        scores['METEOR'] = scores.pop('meteor')
        scores = {'aac_' + key: float(val) for key, val in scores.items()}
        return scores, {}

class AacMetricMeteor(AacMetrics):
    def compute(self, *args, **kwargs):
        scores, info = super().compute(*args, **kwargs)
        scores = {
            'aac_METEOR': scores['aac_METEOR'],
        }
        return scores, info
