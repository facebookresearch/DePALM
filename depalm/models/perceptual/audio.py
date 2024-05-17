# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import laion_clap
from laion_clap.training.data import get_audio_features
import torch
import torch.nn.functional as F

from ..extractors import FeatExtractorWrapper
from ...utils.utility import var_with_shapes

def load_clap_model(model_cfg, config):
    model = laion_clap.CLAP_Module(enable_fusion=model_cfg.fusion)
    model.load_ckpt()
    return model, None


class CLAPExtractor(FeatExtractorWrapper):
    N_FAKE_STATES = 8
    def __init__(self, base_module, norm=None, features_batch_axis=0):
        super().__init__(base_module, norm=norm, features_batch_axis=features_batch_axis)
        self.hidden_states = [self for _ in range(self.N_FAKE_STATES)]

    def _get_audio_embedding_from_data(self, x):
        # From https://github.com/LAION-AI/CLAP/blob/6b1b4b5b4b87f4e19d3836d2ae7d7272e1c69410/src/laion_clap/hook.py#L157
        audio_input = []
        for audio_waveform in x:
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000,
                data_truncating='fusion' if self.base_module.enable_fusion else 'rand_trunc',
                data_filling='repeatpad',
                audio_cfg=self.base_module.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)
        audio_embed = self._clap_model_get_audio_embedding(audio_input)
        return audio_embed

    def _clap_model_get_audio_embedding(self, data):
        # Code from https://github.com/LAION-AI/CLAP/blob/6b1b4b5b4b87f4e19d3836d2ae7d7272e1c69410/src/laion_clap/clap_module/model.py#L720
        # Patch 1: return embedings of ALL tokens
        model = self.base_module.model

        device = next(model.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        #####  [BEGIN PATCH 1]  #####
        audio_embeds = model.encode_audio(input_dict, device=device)
        audio_embeds = torch.cat([audio_embeds["embedding"].unsqueeze(1), audio_embeds["fine_grained_embedding"]], dim=1)
        #####  [END PATCH 1]  #####
        audio_embeds = model.audio_projection(audio_embeds)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

    def forward(self, audio_input):
        with torch.autocast(device_type='cuda', dtype=torch.float32): # Fix nan output values
            features = self._get_audio_embedding_from_data(audio_input)
        self.collected_feats = features
        return torch.stack([features for _ in range(self.N_FAKE_STATES)])

    def get_feature_shape(self):
        raise NotImplementedError
