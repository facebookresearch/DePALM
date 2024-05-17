# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch
import torch.nn as nn

from ...utils.utility import get_activation, get_norm_layer
from .common import TransformerFeatureExtractor, VAR_NUM_TOKENS


class RandomTokenSelection(nn.Module):
    def forward(self, tokens):
        batch_size, n_tokens, _ = tokens.shape
        selected_ids = torch.randint(0, n_tokens, (batch_size,))
        return tokens[torch.arange(batch_size), selected_ids]

class RandomAnyTokenSelection(nn.Module):
    def __init__(self, resampler, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, tokens):
        if self.cfg.eval_full and not self.training:
            return tokens

        frac_tokens = np.random.normal(self.cfg.mean, self.cfg.std)
        frac_tokens = min(self.cfg.max, max(frac_tokens, self.cfg.min))
        if random.random() < self.cfg.full_proba or not self.training:
            frac_tokens = self.cfg.max

        batch_size, n_tokens, _ = tokens.shape
        n_keep_tokens = round(frac_tokens * n_tokens)
        n_keep_tokens = min(n_tokens, max(n_keep_tokens, 1))
        selected_ids = torch.randperm(n_tokens)[:n_keep_tokens]
        selected_ids = selected_ids.sort().values

        tokens = tokens[:, selected_ids]
        return tokens


class TokensResampler(nn.Module):
    def __init__(self, config, n_tokens, embed_dim, adapter_out_dim):
        super().__init__()

        self.config = config
        self.cfg = config.resampler
        self.n_out_tokens = n_tokens # Will change later
        self.embed_dim = embed_dim
        self.out_dim = embed_dim
        self.get_activation = lambda : get_activation(self.config.adapter.activation)

        if self.cfg.type == 'linear' and not config.adapter.transformer.n_layers and not config.adapter.mlp.n_layers:
            self.out_dim = adapter_out_dim
        self.cls_proj_only = None
        if self.out_dim != embed_dim:
            self.cls_proj_only = nn.Linear(embed_dim, self.out_dim)

        # Observe reduction dims
        assert len(config.feat_model.tokens_grid) in [1, 2]
        assert np.prod(config.feat_model.tokens_grid) == n_tokens-1

        if len(config.feat_model.tokens_grid) == 1:
            self.dim_reduction = self.cfg.reduction_fact
        elif len(config.feat_model.tokens_grid) == 2:
            self.dim_reduction = int(self.cfg.reduction_fact**0.5)
            assert self.dim_reduction**2 == self.cfg.reduction_fact, f"reduction factor ({self.cfg.reduction_fact}) should be a squere number in 2D"

        ##### Build the resampler ####
        self._build_proj_module()
        self._build_pool_module()
        self.norm_layer = None
        if self.cfg.normalize:
            self.norm_layer = get_norm_layer(config.adapter.norm_layer, dim=self.out_dim)

        ##### Compute output tokens ####
        if self.cfg.type == 'rand_any_patch':
            self.n_out_tokens = VAR_NUM_TOKENS()
        else:
            self.n_out_tokens -= 1 # Remove CLS token
            assert self.n_out_tokens % self.cfg.reduction_fact == 0
            self.n_out_tokens //= self.cfg.reduction_fact
            if self.cfg.type == 'qpmapper':
                self.n_out_tokens *= self.cfg.qpmapper.n_tokens

            if not self.cfg.drop_cls:
                self.n_out_tokens += self.cfg.duplicate_cls

    def _build_proj_module(self):
        self.in_proj = nn.Sequential()
        if self.cfg.mlp.n_layers:
            for i_layer in range(self.cfg.mlp.n_layers):
                if (i_layer == 0 and self.cfg.mlp.pre_activation) or (i_layer > 0 and self.cfg.mlp.activation):
                    self.in_proj.append(self.get_activation())
                self.in_proj.append(nn.Linear(self.embed_dim, self.embed_dim))

            if self.cfg.mlp.post_activation:
                self.in_proj.append(self.get_activation())


    def _build_pool_module(self):
        if self.cfg.type == 'avg_pool':
            self.pooling = nn.AvgPool1d(self.cfg.reduction_fact)
        elif self.cfg.type == 'max_pool':
            self.pooling = nn.MaxPool1d(self.cfg.reduction_fact)
        elif self.cfg.type == 'conv':
            self.pooling = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=self.cfg.reduction_fact, stride=self.cfg.reduction_fact)
        elif self.cfg.type == 'linear': # Should be equivalent to current "conv"
            self.pooling = nn.Linear(self.embed_dim * self.cfg.reduction_fact, self.out_dim)
        elif self.cfg.type == 'rand_patch':
            self.pooling = RandomTokenSelection()
        elif self.cfg.type == 'rand_any_patch':
            self.pooling = RandomAnyTokenSelection(self, self.cfg.rand_scale)
        elif self.cfg.type == 'qpmapper':
            self.pooling = TransformerFeatureExtractor(
                embed_dim=self.embed_dim,
                num_heads=self.cfg.qpmapper.num_heads,
                num_queries=self.cfg.qpmapper.n_tokens,
                num_layers=self.cfg.qpmapper.n_layers,
                dropout=self.cfg.qpmapper.dropout,
                bias=self.cfg.qpmapper.bias,
                activation=self.config.adapter.activation,
                embed_queries=True,
                is_causal=False,
                n_input_tokens=self.cfg.reduction_fact,
            )
        else:
            raise ValueError(f"Unknown resampler type {self.cfg.type}")

    def forward(self, tokens):
        assert len(tokens.shape) == 3 # (batch_size, n_tokens, embed_dim)
        batch_size, n_tokens, embed_dim = tokens.shape

        cls_token, tokens = tokens[:,:1,:], tokens[:,1:,:]
        n_tokens -= 1

        # Regroup tokens and reshape to (batch_size, self.n_out_tokens (-1), self.cfg.reduction_fact, embed_dim)
        if len(self.config.feat_model.tokens_grid) == 1:
            tokens = tokens.reshape((batch_size, n_tokens//self.dim_reduction, self.dim_reduction, embed_dim,))
        if len(self.config.feat_model.tokens_grid) == 2:
            grid_dim1, grid_dim2 = self.config.feat_model.tokens_grid
            tokens = (tokens
                .reshape((batch_size, grid_dim1//self.dim_reduction, self.dim_reduction, grid_dim2//self.dim_reduction, self.dim_reduction, embed_dim))
                .transpose(2, 3)
                .reshape((batch_size, grid_dim1*grid_dim2//self.cfg.reduction_fact, self.cfg.reduction_fact, embed_dim))
            )
        _, n_tokens_groups, token_group_size, _ = tokens.shape
        tokens = tokens.reshape(batch_size*n_tokens_groups, token_group_size, embed_dim) # Group by token group in standard format

        # Project tokens
        del n_tokens # No longer valid, should not use afterward
        tokens = self.in_proj(tokens)
        if self.cfg.project_cls:
            cls_token = self.in_proj(cls_token)
        if self.cls_proj_only is not None:
            cls_token = self.cls_proj_only(cls_token)

        # Reduce the number of tokens
        if self.cfg.type == 'linear':
            tokens = tokens.reshape(batch_size*n_tokens_groups, 1, token_group_size * embed_dim)
        if self.cfg.type == 'conv':
            tokens = tokens.transpose(1, 2) # Shape (batch_size, channel, sequence)
        if self.cfg.type == 'rand_any_patch':
            assert self.dim_reduction == 1 # Don't use this parameter
            tokens = tokens.reshape(batch_size, n_tokens_groups * token_group_size, embed_dim)
        tokens = self.pooling(tokens)
        if self.cfg.type == 'conv':
            tokens = tokens.transpose(1, 2)

        # Regroup tokens
        tokens = tokens.reshape(batch_size, -1, self.out_dim) # Gather all tokens from all groups tokens together
        if not self.cfg.drop_cls:
            cls_token = cls_token.repeat(1, self.cfg.duplicate_cls, 1)
            tokens = torch.cat([cls_token, tokens], dim=1)

        if self.norm_layer is not None:
            tokens = self.norm_layer(tokens)

        if self.n_out_tokens != VAR_NUM_TOKENS:
            assert tokens.shape[1] == self.n_out_tokens, f"Got {tokens.shape[1]} output tokens instead of {self.n_out_tokens} as predicted" # Safety check
        else:
            self.n_out_tokens.last_n_tokens = tokens.shape[1]
        return tokens
