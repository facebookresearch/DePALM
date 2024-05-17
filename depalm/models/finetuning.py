# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..utils.utility import ModuleWrapper


def apply_prompt_tuning(prompt_cfg, model_text):
    from ..models.adapters.prefix_tokens import LearnedPrefixTokensOnLayer, InsertedTokenPipelinePiece, LearnedPrefixTokensL0

    if prompt_cfg.full_model:
        model_text.wrap_blocks(lambda i_layer, n_layers, layer :
            LearnedPrefixTokensOnLayer(
                layer,
                n_new_tokens=prompt_cfg.new_tokens,
                insert_at=prompt_cfg.insert_at,
                remove_tokens_from_output=True,
                token_norm=False,
                gating=True,
                padding_mode=prompt_cfg.padding_mode,
            )
        )
    else:
        assert prompt_cfg.insert_at == 0, "Should use older method instead (to fix in code)"
        model_text.set_decoder(LearnedPrefixTokensL0(
            model_text.decoder,
            n_new_tokens=prompt_cfg.new_tokens,
        ))

class BiasTuner(torch.nn.Module):
    def __init__(self, tune_shape, bias=True, scaling=True, init_rand=False):
        super().__init__()
        assert len(tune_shape) >= 1
        self.bias = bias
        self.scaling = scaling

        self.bias_param = None
        self.scale_param = None
        if self.bias:
            self.bias_param = torch.nn.Parameter(torch.zeros(tune_shape))
            if init_rand:
                torch.nn.init.normal_(self.bias_param)
        if self.scaling:
            self.scaling_param = torch.nn.Parameter(torch.ones(tune_shape))
            if init_rand:
                torch.nn.init.normal_(self.scaling_param)

    def forward(self, tokens):
        if self.scaling:
            tokens = tokens * self.scaling_param
        if self.bias:
            tokens = tokens + self.bias_param
        return tokens

    @classmethod
    def from_config(cls, tune_shape, cfg):
        assert cfg.enable
        return cls(
            tune_shape,
            bias=cfg.bias,
            scaling=cfg.scaling,
            init_rand=cfg.init_rand,
        )

class BiasTunerWrapper(ModuleWrapper):
    def __init__(self, base_module, bias_tuner):
        super().__init__(base_module)
        self.bias_tuner = bias_tuner

    def forward(self, tokens):
        tokens = self.base_module(tokens)
        tokens = self.bias_tuner(tokens)
        return tokens

    @classmethod
    def wrap_linears(cls, module, cfg):
        if cfg.enable:
            for child_name, child in list(module.named_children()):
                cls.wrap_linears(child, cfg)
                if isinstance(child, torch.nn.Linear):
                    setattr(module, child_name, cls(
                        child, BiasTuner.from_config((child.out_features, ), cfg)
                    ))
