# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from .data_base import load_dataset_from_config
from .vis_datasets import CocoKarpathyDataset, TextcapsDataset, TextVQADataset
from .vqa import VQADataset, OkVqaDataset, AOkVqaDataset
from .audiocaps import AudioCapsDataset
from .video import MSRVTTCaptionDataset


DATASETS = { # Each function should return (data_loaders, task_evaluator)
    'VQA': VQADataset,
    'COCO_karpathy': CocoKarpathyDataset,
    'OKVQA': OkVqaDataset,
    'AOKVQA': AOkVqaDataset,
    'textcaps': TextcapsDataset,
    'text_vqa': TextVQADataset,
    'audiocaps': AudioCapsDataset,
    'msrvtt': MSRVTTCaptionDataset,
}
