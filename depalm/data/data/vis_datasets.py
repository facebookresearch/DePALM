# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, List, Tuple
import random
import json

from PIL import Image
from datasets import load_dataset

from .data_base import FeaturesDataset, DataRow, DataRowMultipleOutputs, compute_cut


class CocoKarpathyDataset(FeaturesDataset):
    """ https://textvqa.org/textcaps/dataset/ """
    DATAFILE = "dataset_coco.json"

    def _preload_dataset(self):
        """ This function should return a list of unpopulated DataRow """
        coco_data = json.load((Path(self.root) / self.DATAFILE).open())
        datarows = []
        for row in coco_data['images']:
            if row['split'] == self.split.value:
                raw_outs = [o['raw'] for o in row['sentences']]
                datarows.append(DataRowMultipleOutputs(
                    input_text="",
                    features_path=str(Path(self.root) / row['filepath'] / row['filename']),
                    output_text_list=raw_outs,
                ))
        return datarows

    def populate_data_row(self, data_row):
        data_row.input_features = Image.open(data_row.features_path).convert('RGB')
        super().populate_data_row(data_row)


class TextcapsDataset(FeaturesDataset):
    """ https://textvqa.org/textcaps/dataset/ """
    LOADED = None
    DATASET_HF = "HuggingFaceM4/TextCaps"

    def _preload_dataset(self):
        """ This function should return a list of unpopulated DataRow """
        if self.LOADED is None:
            self.__class__.LOADED = load_dataset(self.DATASET_HF, cache_dir=self.root)
        split = {
            self.Split.TEST: "test",
            self.Split.TRAIN: "train",
            self.Split.VAL: "validation",
        }[self.split]

        data = self.LOADED[split]
        self.idx_map = list(range(len(data)))
        random.shuffle(self.idx_map)
        return data

    def __len__(self):
        return len(self.idx_map)

    def _row_to_datarow(self, raw_row):
        return DataRowMultipleOutputs(
            input_features=raw_row['image'],
            output_text_list=raw_row['reference_strs'],
        )

    def __getitem__(self, index: int) -> Tuple[Image.Image, Any]:
        row = self.data[self.idx_map[index]]
        data_row = self._row_to_datarow(row)
        self.populate_data_row(data_row)
        self.apply_transforms(data_row)
        return data_row

    def _limit_dataset_size(self, data):
        """ Takes only a part of the training dataset"""
        max_rows = compute_cut(self.max_rows, len(self))
        if max_rows < len(self):
            self.logger.info(f"Using only {max_rows}/{len(self)} rows")
            self.idx_map = self.idx_map[:max_rows]
        return data

class TextVQADataset(TextcapsDataset):
    DATASET_HF = "textvqa"

    def _row_to_datarow(self, raw_row):
        return DataRowMultipleOutputs(
            input_features=raw_row['image'],
            input_text=raw_row['question'],
            output_text_list=raw_row['answers'],
        )