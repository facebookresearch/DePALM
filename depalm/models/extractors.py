# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..utils.utility import ModuleWrapper, unwrap, var_with_shapes, GlobalState
from .perceptual.vision import DisabledModel


class RegisteredHiddenState:
    def __init__(self, state=None):
        self.collected_feats = state

    def store(self, state):
        self.collected_feats = state

    @staticmethod
    def store_row(hidden_state_row, state_values_row):
        assert len(hidden_state_row) == len(state_values_row), f"Got {len(state_values_row)} hidden values, instead of expected {len(hidden_state_row)}"
        for hd, val in zip(hidden_state_row, state_values_row):
            hd.store(val)


class FeatExtractLayerWrapper(ModuleWrapper):
    def __init__(self, base_module, norm=None, features_batch_axis=0):
        super().__init__(base_module)
        self.collected_feats = None
        self.norm = norm
        self.features_batch_axis = features_batch_axis

    def forward(self, *args, **kwargs):
        feats = super().forward(*args, **kwargs)

        self.collected_feats = feats
        if isinstance(self.collected_feats, tuple): # transformers
            assert len(self.collected_feats) == 1
            self.collected_feats = self.collected_feats[0]

        for i_axis in range(self.features_batch_axis, 0, -1):
            self.collected_feats = torch.swapaxes(self.collected_feats, i_axis-1, i_axis)
        if self.norm is not None:
            self.collected_feats = self.norm(self.collected_feats)

        return feats

    def get_feature_shape(self):
        raise NotImplementedError


class FeatExtractorWrapper(ModuleWrapper):
    def __init__(self, base_module, norm, features_batch_axis=0):
        super().__init__(base_module)

        self.norm = norm
        if norm == 'auto':
            unw_model = unwrap(self)
            if hasattr(unw_model, 'norm'):
                self.norm = unw_model.norm
            elif hasattr(unw_model, 'post_layernorm'): # transformers
                self.norm = unw_model.post_layernorm
            elif hasattr(unw_model, 'ln_post'): # OpenCLIP
                self.norm = unw_model.ln_post
            elif hasattr(unw_model, 'layernorm'): # AST
                self.norm = unw_model.layernorm
            elif hasattr(unw_model, 'fc_norm_a'): # MAViL
                self.norm = unw_model.fc_norm_a
            elif isinstance(unw_model, DisabledModel): # Disabled model
                self.norm = None
            else:
                raise NotImplementedError(f"Can't get norm layer of model {base_module}")
        self.features_batch_axis = features_batch_axis
        self.hidden_states = []

    def register_hidden_state(self, hidden_state_wrapper):
        self.hidden_states.append(hidden_state_wrapper)

    def get_blocks(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
        features = torch.stack([hid.collected_feats for hid in self.hidden_states])
        return features


# ========== Model-specific extractors ==========

class DisabledExtractor(FeatExtractorWrapper):
    def __init__(self, base_module, norm=None, **kwargs):
        super().__init__(base_module, norm)
        self.register_hidden_state(base_module)


class VisVitExtractor(FeatExtractorWrapper):
    def __init__(self, base_module, norm=None, features_batch_axis=0):
        super().__init__(base_module, norm, features_batch_axis=features_batch_axis)

        blocks = self.get_blocks()
        for block_id, block in enumerate(blocks):
            blocks[block_id] = self.build_extractor_wrapper(block)
            self.register_hidden_state(blocks[block_id])

        GlobalState.log.info(f"Extracting {len(self.hidden_states)} hidden states")

    def get_blocks(self):
        model = unwrap(self)
        if hasattr(model, 'blocks'): # timm
            return model.blocks
        elif hasattr(model, 'transformer'): # openclip
            return model.transformer.resblocks
        elif hasattr(model, 'encoder'): # transformers
            if hasattr(model.encoder, 'layers'):
                return model.encoder.layers # ViT
            else:
                return model.encoder.layer # AST
        else:
            raise ValueError(f"Can't wrap network {type(model)} with VisVitExtractor")

    def build_extractor_wrapper(self, block):
        return FeatExtractLayerWrapper(block, self.norm, features_batch_axis=self.features_batch_axis)


##### NFNet #####

class NFNetExtractor(FeatExtractorWrapper):
    N_FAKE_STATES = 8
    def __init__(self, base_module, norm=None, features_batch_axis=0):
        super().__init__(base_module, norm=norm, features_batch_axis=features_batch_axis)
        self.hidden_states = [self for _ in range(self.N_FAKE_STATES)]

    def forward(self, *args, **kwargs):
        features = self.base_module.forward(*args, **kwargs)
        fs = features.shape
        features = features.reshape(fs[0], fs[1], fs[2] * fs[3]).transpose(-1, -2)
        self.collected_feats = features
        return torch.stack([features for _ in range(self.N_FAKE_STATES)])

    def get_feature_shape(self):
        raise NotImplementedError

##### DinoV2 #####

class DinoFeatExtractLayerWrapper(FeatExtractLayerWrapper):
    NUM_PATCHES = 256
    def __init__(self, *args, num_features, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features

    def get_feature_shape(self):
        return (self.NUM_PATCHES+1, self.num_features)

    def forward(self, *args, **kwargs):
        feats = super().forward(*args, **kwargs)
        # Invert the patch tokens and cls token to match the format of other vision models
        cls_token, patch_tokens = self.collected_feats[:, :1], self.collected_feats[:, 1:]
        self.collected_feats = torch.concat([cls_token, patch_tokens], dim=1)
        return feats


class DinoV2Extractor(VisVitExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extractor_wrapper(self, block):
        return DinoFeatExtractLayerWrapper(block, self.norm,
            features_batch_axis=self.features_batch_axis,
            num_features=self.base_module.num_features,
        )

##### AST #####

class ASTExtractor(FeatExtractorWrapper):
    N_FAKE_STATES = 8
    def __init__(self, base_module, norm=None, features_batch_axis=0):
        super().__init__(base_module, norm=norm, features_batch_axis=features_batch_axis)
        self.hidden_states = [self for _ in range(self.N_FAKE_STATES)]

    def forward(self, *args, **kwargs):
        features = self.base_module.forward(*args, **kwargs)
        if not isinstance(features, torch.Tensor):
            if isinstance(features, tuple):
                cls_token, patch_tokens = features # MAViL
            else:
                cls_token, patch_tokens = features.pooler_output, features.last_hidden_state
            features = torch.concat([
                cls_token.unsqueeze(1), # "CLS" token
                patch_tokens,
            ], axis=1)
        self.collected_feats = features
        return torch.stack([features for _ in range(self.N_FAKE_STATES)])

    def get_feature_shape(self):
        raise NotImplementedError

class MavilExtractor(FeatExtractorWrapper):
    def __init__(self, base_module, norm=None, features_batch_axis=0):
        super().__init__(base_module, norm=norm, features_batch_axis=features_batch_axis)
        self.hidden_states = [RegisteredHiddenState() for _ in range(12)]

    def forward(self, *args, **kwargs):
        features = self.base_module.forward(*args, **kwargs, all_features=True)
        RegisteredHiddenState.store_row(self.hidden_states, features)
        return features

    def get_feature_shape(self):
        raise NotImplementedError
