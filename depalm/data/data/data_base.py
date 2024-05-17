# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List, Optional, Tuple, Literal
from dataclasses import dataclass, fields, field
from copy import copy
from enum import Enum
import logging
import random
from collections import defaultdict

from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as transforms_lib

from .augmentation.randomized import RandomAugment


##### Base dataclasses for rows #####

@dataclass
class DataRow:
    input_text: str = ""
    input_features: Any = None
    original_features: Any = None # A copy of input_features before applying any transform / data augmentation
    features_path: str = None # Path where to find input_features. Can be used by the dataloaders
    output_text: str = None
    instruction: str = ""

    def copy(self):
        return copy(self)

    def detach_features(self):
        """ Returns a copy without the features, that can be saved in memory. """
        light_row = self.copy()
        light_row.input_features = None
        light_row.original_features = None
        return light_row

    def is_batched(self):
        return isinstance(self.input_text, list)

    def get_single_output_text(self):
        return self.output_text

    def get_outputs(self):
        return [self.output_text]

    @staticmethod
    def stack_rows(rows):
        assert not rows[0].is_batched()
        aggregate_row = rows[0].__class__()
        for field in fields(aggregate_row):
            agg_values = [getattr(row, field.name) for row in rows]
            if isinstance(agg_values[0], torch.Tensor):
                agg_values = torch.stack(agg_values)
            setattr(aggregate_row, field.name, agg_values)
        return aggregate_row

    def unstack(self):
        """ Split a stacked batch of DataRow into individual data rows """
        assert self.is_batched()
        rows = [self.__class__() for _ in range(len(self.input_text))]

        for field in fields(self):
            for row, val in zip(rows, getattr(self, field.name)):
                setattr(row, field.name, val)
        return rows

@dataclass
class DataRowMultipleOutputs(DataRow):
    output_text: Optional[str] = None
    output_text_list: List[str] = field(default_factory=list)

    def get_single_output_text(self):
        if self.is_batched() and self.output_text[0] is None:
            return [random.choice(vals) for vals in self.output_text_list]
        elif self.output_text is None:
            return random.choice(self.output_text_list)
        return super().get_single_output_text()

    def get_outputs(self):
        return self.output_text_list

@dataclass
class DataRowQASortedMO(DataRowMultipleOutputs):
    @staticmethod
    def sort_outputs(outputs):
        counts = defaultdict(lambda : 0)
        for o in outputs:
            counts[o] += 1
        outputs.sort(key=lambda o : counts[o])

    def get_single_output_text(self):
        if self.is_batched() and self.output_text[0] is None:
            for vals in self.output_text_list:
                self.sort_outputs(vals)
            return [vals[0] for vals in self.output_text_list]
        elif self.output_text is None:
            self.sort_outputs(self.output_text_list)
            return self.output_text_list[0]
        return super().get_single_output_text()


##### Dataset classes #####

class DataSplit(Enum):
    TEST = "test"
    TRAIN = "train"
    VAL = "val"

class DataSplitTrainVal(Enum):
    TRAIN = "train"
    VAL = "val"


class FeaturesDataset(Dataset):
    """
        Base class for representing a dataset.
        Iterations should return a DataRow (or a subclass).
    """
    Split = DataSplit

    def __init__(
        self,
        split: "FeaturesDataset.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_rows: Optional[int] = None,
        resize_image_size: Optional[int] = None,
        random_augment: bool = False,
        logger = None,
        instructions: Optional[List[str]] = None,
        instruction_mode: Literal[None, False, 'no', 'default', 'random', 'first'] = None,
        auto_img_transforms=True,
        split_multirow_train=False,
    ) -> None:
        self.root = root
        self.split = split
        self.features_label_transforms = transforms
        self.features_transform = transform
        self.target_transform = target_transform
        self.max_rows = max_rows
        self.instructions = instructions or []
        self.instruction_mode = instruction_mode
        self.random_augment = random_augment
        self.split_multirow_train = split_multirow_train

        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
        self.logger = logger

        # Transforms
        if auto_img_transforms:
            transforms_list = []
            transforms_list.append(transformEnsurePILImage(convert='RGB'))
            if resize_image_size:
                if random_augment:
                    transforms_list.append(transforms_lib.RandomResizedCrop(resize_image_size, scale=(0.5, 1.0), interpolation=Image.BICUBIC, antialias=True))
                else:
                    transforms_list.append(transforms_lib.Resize((resize_image_size, resize_image_size), interpolation=Image.BICUBIC, antialias=True))

            if random_augment:
                transforms_list.append(transformEnsurePILImage())
                transforms_list.append(transforms_lib.RandomHorizontalFlip())
                transforms_list.append(RandomAugment(2, 7, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']))
            transforms_list.append(transformEnsurePILImage())
            if self.features_transform is not None:
                transforms_list.append(self.features_transform)
            transforms_list.append(transformEnsureTensor())
            self.features_transform = transforms_lib.Compose(transforms_list)


        # Pre-load dataset
        self.data = self._preload_dataset()
        self._split_multirows()
        self.logger.info(f"Loaded {len(self)} rows from data split {self.split.value}")
        self.data = self._limit_dataset_size(self.data)

    def _preload_dataset(self) -> List[DataRow]:
        """ This function should return a list of unpopulated DataRow """
        raise NotImplementedError

    def _split_multirows(self):
        if self.split_multirow_train and self.split.value == "train":
            new_data = []
            for row in self.data:
                assert isinstance(row, DataRowMultipleOutputs)
                for output_txt in row.output_text_list:
                    new_data.append(row.copy())
                    new_data[-1].output_text_list = [output_txt]
            self.data = new_data

    def _limit_dataset_size(self, data: List[DataRow]) -> List[DataRow]:
        """ Takes only a part of the training dataset"""
        random.shuffle(data)
        max_rows = compute_cut(self.max_rows, len(self))
        if max_rows < len(data):
            prev_len = len(data)

            data = data[:int(max_rows)]
            self.logger.info(f"Using only {len(data)}/{prev_len} rows")
        return data

    def populate_data_row(self, data_row) -> None:
        data_row.original_features = data_row.input_features
        if self.instruction_mode is None or self.instruction_mode is False or self.instruction_mode == 'no' or not self.instructions:
            data_row.instruction = ""
        elif self.instruction_mode == 'random':
            data_row.instruction = random.choice(self.instructions)
        elif self.instruction_mode == 'first':
            data_row.instruction = self.instructions[0]
        elif isinstance(self.instruction_mode, int):
            data_row.instruction = self.instructions[self.instruction_mode]
        elif self.instruction_mode == 'default':
            pass # Use the instruction from the dataset if it exists
        else:
            raise ValueError(f"Unkown instruction_mode {self.instruction_mode}")

        if not '{prompt}' in data_row.instruction:
            data_row.instruction = data_row.instruction + ' {prompt}'
        data_row.instruction = data_row.instruction.strip()

    def apply_transforms(self, data_row) -> None:
        if self.features_transform is not None:
            data_row.input_features = self.features_transform(data_row.input_features)
        if self.target_transform is not None:
            if data_row.output_text is not None:
                data_row.output_text = self.target_transform(data_row.output_text)
            if isinstance(data_row, DataRowMultipleOutputs):
                data_row.output_text_list = list(map(self.target_transform, data_row.output_text_list))
        if self.features_label_transforms is not None:
            if isinstance(data_row, DataRowMultipleOutputs):
                raise NotImplementedError
            data_row.input_features, data_row.output_text = self.features_label_transforms(data_row.input_features, data_row.output_text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> DataRow:
        data_row = self.data[index].copy()
        self.populate_data_row(data_row)
        self.apply_transforms(data_row)
        return data_row

    def collate_fn(self, batch: List[DataRow]) -> DataRow:
        """ Returns a single DataRow where each field is an aggregate """
        return DataRow.stack_rows(batch)

    @classmethod
    def from_config(cls, config: OmegaConf, split, **kwargs):
        return cls(
            split=split,
            resize_image_size=config.resize_image_size,
            random_augment=config.splits[split.value].random_augment,
            instructions=config.instructions,
            instruction_mode=config.insert_instruction,
            split_multirow_train=config.split_multirow_train,
            **kwargs
        )


def load_dataset_from_config(dataset_cls, dataset_config, accelerator, feat_transform=None):
    """ Function used on this codebase only, to convert from an OmegaConf object to a dataset """
    loaders = {}

    for split in dataset_cls.Split:
        split_config = dataset_config.splits[split.value]
        dataset = dataset_cls.from_config(
            dataset_config,
            split=split,
            root=dataset_config.root,
            max_rows=split_config.max_rows,
            transform=feat_transform,
            logger=accelerator.log,
        )

        sampler = None
        if accelerator.distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                num_replicas=accelerator.world_size,
                rank=accelerator.local_rank,
                drop_last=True,
            )

        loaders[split.value] = DataLoader(
            dataset,
            batch_size=split_config.batch_size,
            shuffle=None if (sampler is not None) else False,
            num_workers=accelerator.num_workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

    return loaders


##### Utilities #####

def compute_cut(max_rows, n_rows, strict=False):
    if max_rows is None:
        return n_rows
    elif 0 < max_rows < 1:
        return round(max_rows * n_rows)
    else:
        assert isinstance(max_rows, int), f"{max_rows=} should be an integer"
        if not strict:
            max_rows = min(max_rows, n_rows)
        assert 0 <= max_rows <= n_rows, f"{max_rows=} should be between 0 and {n_rows}"
        return max_rows

class transformEnsureTensor(Callable):
    def __init__(self):
        self.to_tensor = transforms_lib.ToTensor()

    def __call__(self, sample):
        return sample if isinstance(sample, torch.Tensor) else self.to_tensor(sample)

class transformLog(Callable):
    def __init__(self, name='transformLog'):
        self.name = name

    def __call__(self, sample):
        from depalm.utils.utility import var_with_shapes
        print(self.name, 'sample ->', var_with_shapes(sample))
        return sample

class transformEnsurePILImage(Callable):
    def __init__(self, convert=None):
        self.convert = convert
        self.to_pil = transforms_lib.ToPILImage()

    def __call__(self, sample):
        if not isinstance(sample, Image.Image):
            sample = torch.tensor(sample).float()
            sample = self.to_pil(sample)
        if hasattr(self, 'convert') and self.convert:
            sample = sample.convert(self.convert)
        return sample
