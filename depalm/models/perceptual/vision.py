# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Callable
import requests
import time

import torch
import timm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, ASTModel
import torchvision.transforms as transforms_lib
from huggingface_hub import hf_hub_download
import open_clip

from ...utils.utility import GlobalState
from ...data.data.data_base import transformEnsureTensor, transformEnsurePILImage


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def preload_hf_model(repo_id, filename, cache_dir) -> str:
    dest_file = Path(cache_dir) / repo_id / filename
    if not dest_file.exists():
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dl_path = hf_hub_download(repo_id, filename, cache_dir=cache_dir, local_dir_use_symlinks=False)
        print("Got dl_path", dl_path)
        os.symlink(dl_path, str(dest_file))
    return str(dest_file)

def autoload_timm_model(model_cfg, config):
    model = timm.create_model(
        model_cfg.name,
        pretrained=True,
        pretrained_cfg_overlay=dict(file=model_cfg.data_path),
        global_pool='',
        num_classes=0,
    )
    GlobalState.log.info(f"Loading pretrained {model_cfg.name}")

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    return model, transforms

def autoload_openclip_model(model_cfg, config):
    model, _, transforms = open_clip.create_model_and_transforms(
        f'hf-hub:{model_cfg.name}',
        cache_dir=model_cfg.data_path,
    )
    model = model.visual # Keep only the visual encoder

    return model, transforms

class transformFromProcessor(Callable):
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, sample):
        t_sample = self.processor(images=sample, return_tensors="pt")['pixel_values']
        assert t_sample.shape[0] == 1 # The processor adds a dim
        t_sample = t_sample[0]
        return t_sample

def autoload_transformers_model(model_cfg, config):
    processor = AutoProcessor.from_pretrained(model_cfg.name)
    model = AutoModelForZeroShotImageClassification.from_pretrained(model_cfg.name)
    model = model.vision_model
    transforms = transforms_lib.Compose([
        transformEnsurePILImage(),
        transformFromProcessor(processor),
    ])
    return model, transforms

def load_dinov2(model_cfg, config):
    torch.hub.set_dir(model_cfg.data_path)
    model = torch.hub.load('facebookresearch/dinov2', model_cfg.name)

    transforms = transforms_lib.Compose([
        transforms_lib.Resize(model_cfg.resize_size, interpolation=transforms_lib.InterpolationMode.BICUBIC),
        transforms_lib.CenterCrop(model_cfg.crop_size),
        transformEnsureTensor(),
        transforms_lib.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    return model, transforms

def load_ast_model(model_cfg, config):
    return ASTModel.from_pretrained(model_cfg.name), None

def load_dummy_disabled_model(model_cfg, config):
    return DisabledModel(), None

class DisabledModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_tensor = torch.ones((1, 1))
        self.collected_feats = None

    def forward(self, tensor):
        self.collected_feats = torch.ones((len(tensor), 1, 1), device=tensor.device, dtype=tensor.dtype)
        return self.collected_feats

    def get_feature_shape(self):
        return self.dummy_tensor.shape