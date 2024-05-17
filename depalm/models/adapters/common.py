# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Any
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from ...utils.utility import get_activation, get_norm_layer, var_with_shapes, GlobalState


class VAR_NUM_TOKENS:
    def __init__(self):
        self.last_n_tokens = None

    def __eq__(self, __value: object) -> bool:
        if __value == VAR_NUM_TOKENS or __value is self:
            return True
        return False

    def __int__(self):
        assert self.last_n_tokens is not None
        return self.last_n_tokens

    def __add__(self, o):
        return int(self) + o


class VAR_NUM_TOKENS_LAMBDA(VAR_NUM_TOKENS):
    def __init__(self, lbd):
        self.lbd = lbd

    @property
    def last_n_tokens(self):
        try:
            last_n_tokens = self.lbd()
            if isinstance(last_n_tokens, VAR_NUM_TOKENS):
                return last_n_tokens.last_n_tokens
        except Exception as e:
            print('VAR_NUM_TOKENS_LAMBDA', e)
            print(traceback.format_exc())
            raise e

# ========== Apply on models ==========

def apply_adapter_transformers(adapter_cls):
    def builder(config, model_text, feat_model):
        model_text.set_decoder(adapter_cls(config, model_text.decoder, feat_model))
        return model_text
    return builder


# ========== Modules ==========

def inner_padding_2d_attention_mask(in_tensor, paddings, att_value, no_att_value, padding_mode):
    """
        Insert row / columns with a constant value (att_value or no_att_value depending of the position)

        paddings: list / tuple of pairs (insert_at, length). The first is for the dimension -2, the second for -1

        padding mods:
        - causal (diagonal matrix)
        - non-causal (causal, but tokens of the block know each other)
        - prev-non-causal (causal, but tokens of the block and before know each other)
        - zero (padd with 0 only, no causality)
    """
    assert padding_mode in ['causal', 'non-causal', 'prev-non-causal', 'zero']

    if padding_mode == 'zero':
        tri_matrix = torch.zeros(in_tensor.shape[-2] + paddings[0][1], in_tensor.shape[-1] + paddings[1][1], dtype=in_tensor.dtype, device=in_tensor.device)
    else:
        tri_matrix = torch.triu(
            torch.ones(in_tensor.shape[-2] + paddings[0][1], in_tensor.shape[-1] + paddings[1][1], dtype=in_tensor.dtype, device=in_tensor.device),
            diagonal=1
        )
        if padding_mode == 'non-causal': # "auto-attention"
            tri_matrix[paddings[0][0]:paddings[0][0]+paddings[0][1], paddings[1][0]:paddings[1][0]+paddings[1][1]] = 0.0

    tri_matrix = tri_matrix * no_att_value + (1 - tri_matrix) * att_value
    tri_matrix = tri_matrix.reshape((1, 1) + tri_matrix.shape).expand(in_tensor.shape[:-2] + tri_matrix.shape)

    # Insert in first dim
    insert_at, length = paddings[0]
    in_tensor = torch.cat((
        in_tensor[..., :insert_at, :],
        torch.cat([
            tri_matrix[..., insert_at:insert_at+length, :paddings[1][0]],
            tri_matrix[..., insert_at:insert_at+length, paddings[1][0] + paddings[1][1]:],
        ], dim=-1),
        in_tensor[..., insert_at:, :],
    ), dim=-2)

    # Insert at second dim
    insert_at, length = paddings[1]
    in_tensor = torch.cat((
        in_tensor[..., :insert_at],
        tri_matrix[..., insert_at:insert_at+length],
        in_tensor[..., insert_at:],
    ), dim=-1)

    if padding_mode == 'prev-non-causal': # full-prev-auto-attention?
        in_tensor[..., :paddings[0][0]+paddings[0][1], :paddings[1][0]+paddings[1][1]] = att_value

    return in_tensor


class SequentialConnector(nn.Sequential):
    def __init__(self, *args, features_ids, prefixs_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_ids = features_ids
        self.prefixs_ids = prefixs_ids

    def forward(self, hiden_feats: torch.Tensor, hidden_state_id, prefix_adapter_id) -> torch.Tensor:
        local_hid_id = self.features_ids.index(hidden_state_id)
        local_prefix_adapt_id = self.prefixs_ids.index(prefix_adapter_id)
        for module in self:
            if isinstance(module, (TransformerFeatureExtractor, QFormerFeatureExtractor)):
                hiden_feats = module(
                    hiden_feats,
                    local_hid_id=local_hid_id,
                    local_prefix_adapt_id=local_prefix_adapt_id,
                )
            else:
                hiden_feats = module(hiden_feats)
        return hiden_feats


class FCMLP(nn.Module):
    def __init__(self, embed_dim, scale_fact, out_dim=None, in_dim=None, n_layers=2, activation='gelu', norm_layer='rms_norm', residual=True):
        super().__init__()
        self.residual = residual
        if out_dim is None:
            out_dim = embed_dim
        if in_dim is None:
            in_dim = embed_dim
        inner_dim = int(embed_dim * scale_fact)
        assert n_layers >= 1

        self.in_proj = None
        if in_dim != out_dim and residual:
            assert out_dim == embed_dim
            self.in_proj = nn.Linear(in_dim, embed_dim)
            in_dim = embed_dim

        self.norm_layer = get_norm_layer(norm_layer, out_dim)

        self.layers = nn.Sequential()
        for i_layer in range(n_layers):
            self.layers.append(nn.Linear(
                in_dim if i_layer == 0 else inner_dim,
                out_dim if i_layer == n_layers-1 else inner_dim,
            ))
            if i_layer < n_layers-1:
                self.layers.append(get_activation(activation))

    def forward(self, x):
        if self.in_proj is not None:
            x = self.in_proj(x)
        res_x = x
        x = self.layers(x)
        if self.residual:
            x = res_x + x
        x = self.norm_layer(x)
        return x


class CLSPosEmbeding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        bsz, num_tokens, _ = tokens.size()
        embed = torch.cat([torch.ones(bsz, 1, 1), torch.zeros(bsz, num_tokens-1, 1)], dim=1)
        embed = embed.to(tokens.device)
        return torch.cat([tokens, embed], dim=-1)


class LearnableQueriesAttention(nn.Module):
    # Based on https://github.com/huggingface/transformers/blob/2489e380e45177eb3da108366f65b8108b2815c5/src/transformers/models/opt/modeling_opt.py#L122

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_queries: int,
        dropout: float = 0.0,
        bias: bool = True,
        linear: bool = True,
        out_dim: Optional[int] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.out_dim = out_dim if out_dim is not None else embed_dim

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.query_tokens = torch.nn.Parameter(torch.empty((num_queries, embed_dim)))
        torch.nn.init.normal_(self.query_tokens)
        if linear:
            self.out_proj = nn.Linear(embed_dim, self.out_dim, bias=bias)
        else:
            self.out_proj = nn.Identity()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        bsz, _, _ = hidden_states.size()
        tgt_len = self.num_queries

        # self_attention
        query_states = self._shape(self.query_tokens.repeat(bsz, 1, 1), tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        attn_output = F.scaled_dot_product_attention(
            query=self._shape(query_states, tgt_len, bsz),
            key=key_states,
            value=value_states,
            dropout_p=self.dropout if not self.training else 0,
        ).transpose(1, 2).contiguous().reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerFeatureExtractor(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries, n_input_tokens, num_layers, dropout=0, bias=True, activation="gelu", embed_queries=True, n_prefix_queries=None, is_causal=False, keep_cls_token=False, pos_encoding=None):
        super().__init__()
        assert bias == True, "This implementation doesn't support no bias yet"
        self.num_queries = num_queries
        self.embed_queries = embed_queries
        self.n_prefix_queries = n_prefix_queries # if not None, will use 'n_prefix_queries' different queries vectors
        self.is_causal = is_causal
        self.keep_cls_token = keep_cls_token
        self.pos_encoder = None
        self.n_input_tokens = n_input_tokens

        if self.embed_queries:
            self.query_tokens = torch.nn.Parameter(torch.empty((n_prefix_queries or 1, num_queries, embed_dim)))
            torch.nn.init.normal_(self.query_tokens)

        self.blocks = torch.nn.Sequential(*[
            torch.nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation=get_activation(activation),
                layer_norm_eps=1e-4,
                batch_first=True,
                norm_first=False
            ) for _ in range(num_layers)
        ])

        if pos_encoding:
            self.n_embed_tokens_max = n_input_tokens*2
            if pos_encoding == 'learned':
                self.pos_encoder = nn.Embedding(self.n_embed_tokens_max, embed_dim)
            elif pos_encoding == 'rotary':
                self.pos_encoder = LlamaRotaryEmbedding(embed_dim, self.n_embed_tokens_max)
            else:
                assert pos_encoding in ['CLS', None]

    def forward(self, tokens, local_hid_id: int=None, local_prefix_adapt_id: int=None):
        if self.n_prefix_queries:
            assert local_prefix_adapt_id is not None
        bsz, n_tokens, embed_dim = tokens.shape

        if self.pos_encoder is not None:
            if isinstance(self.pos_encoder, nn.Embedding):
                assert n_tokens <= self.n_embed_tokens_max
                pos_ids = torch.arange(n_tokens, device=tokens.device)
                tokens = tokens + self.pos_encoder(pos_ids)
            else:
                tokens = self.pos_encoder(tokens.view(bsz, 1, n_tokens, embed_dim), n_tokens).view(bsz, n_tokens, embed_dim)

        if self.embed_queries:
            query_id = 0 if self.n_prefix_queries is None else local_prefix_adapt_id
            query_tokens = self.query_tokens[query_id].repeat(bsz, 1, 1)
            tokens = torch.cat([tokens, query_tokens], dim=1)

        mask = None
        if self.is_causal:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(n_tokens, device=tokens.device)
            assert mask.shape == (n_tokens, n_tokens)
            mask = mask.repeat(bsz, 1, 1)
        for block in self.blocks:
            tokens = block(tokens, src_mask=mask, is_causal=self.is_causal)
        if self.embed_queries:
            cls_token = tokens[:,:1,:]
            tokens = tokens[:,-self.num_queries:,:]
            if self.keep_cls_token:
                tokens = torch.cat([cls_token, tokens[:,1:,:]], dim=1)
        elif self.num_queries:
            tokens = tokens[:,:self.num_queries,:]
        return tokens

class QFormerFeatureExtractor(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries, num_layers, dropout=0, bias=True, activation="gelu", n_prefix_queries=None):
        super().__init__()
        assert bias == True, "This implementation doesn't support no bias yet"
        assert num_queries > 0
        self.num_queries = num_queries
        self.n_prefix_queries = n_prefix_queries # if not None, will use 'n_prefix_queries' different queries vectors

        self.query_tokens = torch.nn.Parameter(torch.empty((n_prefix_queries or 1, num_queries, embed_dim)))
        torch.nn.init.normal_(self.query_tokens)

        self.blocks = torch.nn.Sequential(*[
            torch.nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=dropout,
                activation=get_activation(activation),
                layer_norm_eps=1e-4,
                batch_first=True,
                norm_first=False
            ) for _ in range(num_layers)
        ])

    def forward(self, vis_tokens, local_hid_id: int, local_prefix_adapt_id: int):
        bsz, _, _ = vis_tokens.shape
        query_id = 0 if self.n_prefix_queries is None else local_prefix_adapt_id
        tokens = self.query_tokens[query_id].repeat(bsz, 1, 1)

        for block in self.blocks:
            tokens = block(tokens, vis_tokens)

        assert tokens.shape[1] == self.num_queries
        return tokens

class CrossAttentionFeatExtractor(torch.nn.Module):
    def __init__(self, llm_dim, embed_dim, num_heads, dropout=0, gating=True, norm=True, vis_mlp_layers=0, llm_mlp_layers=0, out_mlp_layers=0, activation='gelu', norm_layer='layer_norm', mlp_scale=1, residual=False):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.gating = gating
        self.out_mlp_layers = out_mlp_layers
        self.vis_mlp_layers = vis_mlp_layers
        self.llm_mlp_layers = llm_mlp_layers
        self.residual = False
        self.activation = get_activation(activation)

        self.llm_q_proj = nn.Linear(embed_dim if llm_mlp_layers else llm_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, llm_dim)
        self.norm = get_norm_layer(norm_layer, embed_dim) if norm else nn.Identity()
        if gating:
            assert gating in [True, 'zero', 'scheduled']
            self.gating_value = torch.nn.Parameter(torch.tensor(0.))
            self.tanh = nn.Tanh()

        if out_mlp_layers:
            self.res_mlp = FCMLP(embed_dim, mlp_scale, n_layers=out_mlp_layers, activation=activation, norm_layer=norm_layer, residual=residual)
        if vis_mlp_layers:
            self.vis_res_mlp = FCMLP(embed_dim, mlp_scale, n_layers=vis_mlp_layers, activation=activation, norm_layer=norm_layer, residual=residual)
        if llm_mlp_layers:
            self.llm_res_mlp = FCMLP(embed_dim, mlp_scale, in_dim=llm_dim, n_layers=llm_mlp_layers, activation=activation, norm_layer=norm_layer, residual=residual)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        tensor = tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        return tensor.view((bsz * self.num_heads, seq_len, self.head_dim))

    def forward(self, feat_tokens, llm_tokens) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        res_llm_tokens = llm_tokens
        if self.llm_mlp_layers:
            llm_tokens = self.llm_res_mlp(llm_tokens)
        if self.vis_mlp_layers:
            feat_tokens = self.vis_res_mlp(feat_tokens)

        feat_tokens = feat_tokens.repeat_interleave(len(llm_tokens) // len(feat_tokens), dim=0) # To accomodate for beam search
        bsz, _, _ = llm_tokens.size()
        tgt_len = llm_tokens.shape[1]

        ##### [Begin Cross-attention] #####
        query_states = self._shape(self.llm_q_proj(llm_tokens), -1, bsz)
        key_states = self._shape(self.k_proj(feat_tokens), -1, bsz)
        value_states = self._shape(self.v_proj(feat_tokens), -1, bsz)

        attn_output = F.scaled_dot_product_attention(
            query=self._shape(query_states, tgt_len, bsz),
            key=key_states,
            value=value_states,
            dropout_p=self.dropout if not self.training else 0,
        ).transpose(1, 2).contiguous().reshape(bsz, tgt_len, self.embed_dim)

        ##### [End Cross-attention] #####

        if self.residual:
            tokens = self.norm(attn_output + llm_tokens)
        else:
            tokens = self.norm(attn_output)
        if self.out_mlp_layers:
            tokens = self.res_mlp(tokens)
        tokens = self.out_proj(tokens)

        gating_value = 1
        if self.gating:
            gating_value = self.tanh(self.gating_value)
            if self.gating == 'zero':
                gating_value = gating_value * 0
            elif self.gating == 'scheduled':
                FRAC_SCHEDULE = 0.3
                gating_value = gating_value * min(1, GlobalState.training_fract / FRAC_SCHEDULE)
            else:
                assert self.gating is True
        tokens = res_llm_tokens + gating_value * tokens
        return tokens
