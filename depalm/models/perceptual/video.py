# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import torch
import torch.nn.functional as F
from einops import rearrange

from ..extractors import FeatExtractorWrapper, RegisteredHiddenState


def load_timesformer_model(model_cfg, config):
    from .timesformermodel import TimeSformer

    # FIX CONFIG
    if not model_cfg.average_time and not model_cfg.fixed_time_dims:
        model_cfg.fixed_time_dims = True
        model_cfg.hidden_feature_shape[0] = 1 + (model_cfg.hidden_feature_shape[0]-1) * config.dataset.num_frames
        model_cfg.tokens_grid[0] *= config.dataset.num_frames

    model = TimeSformer(img_size=config.dataset.resize_image_size, num_frames=config.dataset.num_frames, attention_type='divided_space_time', pretrained_model=model_cfg.data_path, return_hidden_state=True, space_only_for_images=None)
    return model, None


class TimeSFormerExtractor(FeatExtractorWrapper):
    def __init__(self, base_module, **kwargs):
        super().__init__(base_module, **kwargs)
        self.hidden_states = [RegisteredHiddenState() for _ in range(12)]
        self.config = self.base_module._global_config

    def forward(self, video_input):
        _, features = self.base_module(video_input)
        if self.config.feat_model.average_time:
            new_features = []
            for f in features:
                cls_token, patch_tokens = f[:,:1], f[:,1:]
                patch_tokens = patch_tokens.reshape(len(video_input), -1, 196, 768)
                patch_tokens = patch_tokens.mean(dim=1)
                f = torch.cat([cls_token, patch_tokens], dim=1)
                new_features.append(f)
            features = new_features

        RegisteredHiddenState.store_row(self.hidden_states, features)
        return features

    def get_feature_shape(self):
        raise NotImplementedError
