# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

import numpy as np
import torch

from ...models.extractors import DisabledExtractor
from ...utils.utility import get_input_feature_of, ModuleWrapper, sublist_ids_from_cfg, DropFeaturesLayer, unwrap, var_with_shapes, get_activation, get_norm_layer, GlobalState, call_with_splited_batch
from .gating import GatedAttentionLayer
from .common import inner_padding_2d_attention_mask, LearnableQueriesAttention, TransformerFeatureExtractor, QFormerFeatureExtractor, SequentialConnector, CLSPosEmbeding, CrossAttentionFeatExtractor, VAR_NUM_TOKENS, VAR_NUM_TOKENS_LAMBDA
from ..finetuning import BiasTuner
from .resamplers import TokensResampler


class InsertedTokenPipelinePiece(ModuleWrapper):
    """
    This class defines a wrapper around a transformer block that automatically changes the inputs to take care of any inserted token.
    It expands attention mask and position ids, and can remove the inserted token if needed.
    """
    def __init__(self, att_layer, n_new_tokens, remove_tokens_from_output=False, insert_at=0, padding_mode=None):
        super().__init__(att_layer)
        self._base_n_new_tokens = n_new_tokens
        self.remove_tokens_from_output = remove_tokens_from_output
        self.insert_at = insert_at # Where the token was inserted. Should be 1 if there is a prefix token
        self.padding_mode = padding_mode

        if isinstance(self.base_module, InsertedTokenPipelinePiece):
            # Move the interval so that it doesn't intersect with the other one
            # This fix only works with 2 InsertedTokenPipelinePiece, when both inser AND remove on a single layer
            if type(self) == InsertedTokenPipelinePiece or self.base_module.remove_tokens_from_output:
                pass # Nothing to fix then, should work as-is if we want external injection to be AFTER the internal injection, inside the prompt
            else:
                assert not isinstance(self.base_module.base_module, InsertedTokenPipelinePiece)
                assert self.base_module.insert_at == self.insert_at
                self.base_module.insert_at += self.n_new_tokens

    @property
    def n_new_tokens(self):
        if self._base_n_new_tokens == VAR_NUM_TOKENS:
            return self._base_n_new_tokens.last_n_tokens
        return self._base_n_new_tokens

    def remove_inserted_tokens(self, tokens):
        return torch.cat((
            tokens[...,:self.insert_at,:],
            tokens[...,self.insert_at+self.n_new_tokens:,:]
        ), dim=-2)

    def forward(self, tokens, **kwargs):
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None and kwargs['attention_mask'].shape[-1] != tokens.shape[1]:
            # Resize the attention mask
            attention_mask = kwargs['attention_mask']

            assert attention_mask[0,0,0,0] == 0, "Not the type of attention mask we should have"
            if attention_mask.shape[-1] > 1:
                att_value, no_att_value = attention_mask[0,0,0,0], attention_mask[0,0,0,-1]
            else:
                att_value, no_att_value = attention_mask[0,0,0,0], -1e9
            att_value, no_att_value = 0, -65504

            attention_mask = inner_padding_2d_attention_mask(
                attention_mask,
                ((self.insert_at, self.n_new_tokens), (self.insert_at, self.n_new_tokens)),
                att_value, no_att_value,
                padding_mode=self.padding_mode,
            )
            kwargs['attention_mask'] = attention_mask

        if kwargs.get('position_ids', None) is not None:
            kwargs['position_ids'] = torch.cat([
                torch.arange(len(sub_t), device=kwargs['position_ids'].device).unsqueeze(0)
                for sub_t in tokens
            ])
        att_output = super().forward(tokens, **kwargs)
        assert isinstance(att_output, tuple)
        att_output = list(att_output)

        if self.remove_tokens_from_output:
            assert att_output[0].shape[-2] >= self.n_new_tokens
            att_output[0] = self.remove_inserted_tokens(att_output[0])

        return tuple(att_output)


class ABCPrefixTokensOnLayer(InsertedTokenPipelinePiece):
    def __init__(self, att_layer, n_new_tokens, replace_tokens=False, remove_tokens_from_output=False, insert_at=0, token_norm=False, gating=False, padding_mode=None, bias_tuning=None, norm_layer='layer_norm'):
        super().__init__(att_layer, n_new_tokens=n_new_tokens, remove_tokens_from_output=remove_tokens_from_output, insert_at=insert_at, padding_mode=padding_mode)

        self.prompt_dim = get_input_feature_of(att_layer)
        self.replace_tokens = replace_tokens # Should be True when replacing prefix tokens from previous layer
        self.gating_value = torch.tensor(1.)
        self.token_norm = token_norm
        self.adapter_bias_tuning = None

        if self.token_norm:
            self.adapter_norm_layer = get_norm_layer(norm_layer, (self.prompt_dim,))
        if gating:
            base_mod = unwrap(self)
            if hasattr(base_mod, 'self_attn'):
                base_mod.self_attn = GatedAttentionLayer(
                    base_mod.self_attn,
                    from_id=self.insert_at,
                    to_id=self.insert_at+self.n_new_tokens,
                )
            else:
                raise ValueError(f"Can't use gating with layertype {type(base_mod)}")
        if bias_tuning is not None and bias_tuning.enable:
            tune_shape = (self.prompt_dim, )
            if not bias_tuning.share_on_tokens:
                tune_shape = (n_new_tokens, ) + tune_shape
            self.adapter_bias_tuning = BiasTuner.from_config(tune_shape, bias_tuning)

    @abstractmethod
    def get_prompt_tokens(self):
        raise NotImplementedError

    def get_prompt_tokens_shape(self):
        return (self.n_new_tokens, self.prompt_dim)

    def forward(self, tokens, **kwargs):
        use_cache = kwargs.get('use_cache', False)
        if use_cache:
            assert False, "Can't use cache for now"
            assert not self.replace_tokens and self.remove_tokens_from_output, "Can't use cache with tokens inserted "

        if 'past_key_value' in kwargs:
            add_prompt_tokens = kwargs['past_key_value'] is None
        else:
            raise NotImplementedError(f"Can't use insert prefix tokens with this type of layers ({type(unwrap(self))})")
        assert add_prompt_tokens is True, f"Can't manage prefix tokens insertion with cache for now (got past_key_value={var_with_shapes(kwargs['past_key_value'])})"

        if add_prompt_tokens:
            if self.replace_tokens:
                tokens = self.remove_inserted_tokens(tokens)

            prompt_tokens = self.get_prompt_tokens().to(tokens.device) # Get the inserted tokens
            if self.token_norm:
                prompt_tokens = self.adapter_norm_layer(prompt_tokens)
            if self.adapter_bias_tuning:
                prompt_tokens = self.adapter_bias_tuning(prompt_tokens)
            prompt_tokens = prompt_tokens.repeat_interleave(len(tokens) // len(prompt_tokens), dim=0)  # Duplicate tokens accross batch
            tokens = torch.cat((tokens[...,:self.insert_at,:], prompt_tokens, tokens[...,self.insert_at:,:]), dim=-2) # Concatenate tokens to each batch sequence

        att_output = super().forward(tokens, **kwargs)
        assert isinstance(att_output, tuple)
        att_output = list(att_output)

        return tuple(att_output)


class LearnedPrefixTokensOnLayer(ABCPrefixTokensOnLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_tokens = torch.nn.Parameter(torch.empty((1, self.n_new_tokens, self.prompt_dim)))
        torch.nn.init.normal_(self.prompt_tokens)

    def get_prompt_tokens(self):
        return self.prompt_tokens


class PrefixTokensOnLayerWithConnector(ABCPrefixTokensOnLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stored_extracted_feats = None

    def get_prompt_tokens(self):
        assert self._stored_extracted_feats is not None
        return self._stored_extracted_feats

    def store_extracted_feature(self, extracted_feats):
        self._stored_extracted_feats = extracted_feats

class CrossAttentionLayerAdapter(ModuleWrapper):
    def __init__(self, base_module, cross_attn_layer, n_vis_tokens):
        super().__init__(base_module)
        self._cross_attn_layer = (cross_attn_layer, )
        self._stored_extracted_feats = None
        self.prompt_dim = get_input_feature_of(base_module)
        self.n_vis_tokens = n_vis_tokens

    def get_prompt_tokens_shape(self):
        return (self.n_vis_tokens, self.prompt_dim)

    def store_extracted_feature(self, extracted_feats):
        self._stored_extracted_feats = extracted_feats

    def forward(self, tokens, **kwargs):
        tokens = self._cross_attn_layer[0](self._stored_extracted_feats, tokens)
        return super().forward(tokens, **kwargs)


# ========== LLM wrappers ==========

class PrefixTokensInnerLayersAdapter(ModuleWrapper):
    def __init__(self, config, base_module, feat_extractor):
        super().__init__(base_module)

        self.config = config

        # Collect the hidden states references
        if isinstance(feat_extractor, DisabledExtractor):
            hidden_states_ids = [0]
        else:
            hidden_states_ids = sublist_ids_from_cfg(config.adapter.source, n_total=len(feat_extractor.hidden_states))
        self.hidden_states = [feat_extractor.hidden_states[i] for i in hidden_states_ids]

        self.feats_shape = self.config.feat_model.hidden_feature_shape
        if self.feats_shape in [None, 'auto']:
            self.feats_shape = self.hidden_states[0].get_feature_shape()

        self.embed_dim = self.config.adapter.embed_dim
        if self.embed_dim in ['auto', None, 0]:
            self.embed_dim = self.feats_shape[-1]

        # Build the adapters (Wrappers around layers, no parameters)
        self.n_new_tokens = self._compute_n_new_tokens(self.feats_shape)
        self.prefix_adapters = self.build_prefix_adapters()

        if self.prefix_adapters is not None: # Check if we are not a subclass with a custom behaviour
            llm_insert_shape = self.prefix_adapters[0].get_prompt_tokens_shape()

            # Build the connectors (connection layers, can connect multiple hidden states / adapters pairs)
            # n_connectors = config.adapter.sharing.n_connections if config.adapter.sharing.share_connections else len(self.prefix_adapters)
            assert config.adapter.sharing.share_connections in [True, False]
            n_connectors = config.adapter.sharing.n_connections if config.adapter.sharing.share_connections else len(self.hidden_states)
            feats_on_connector = [[] for _ in range(n_connectors)]
            prefix_adapt_on_connector = [[] for _ in range(n_connectors)]

            for prefix_adapt_id in range(len(self.prefix_adapters)):
                hidden_state_id = (prefix_adapt_id * len(self.hidden_states)) // len(self.prefix_adapters)
                connector_id = (prefix_adapt_id * n_connectors) // len(self.prefix_adapters)
                feats_on_connector[connector_id].append(hidden_state_id)
                prefix_adapt_on_connector[connector_id].append(prefix_adapt_id)

            self.adapter_connectors = torch.nn.ModuleList([
                self._build_connector(
                    in_shape=self.feats_shape,
                    out_shape=llm_insert_shape,
                    features_ids=feats_on_connector[connector_id],
                    prefixs_ids=prefix_adapt_on_connector[connector_id],
                )
                for connector_id in range(n_connectors)
            ])

            # Log
            if len(hidden_states_ids) > len(self.prefix_adapters):
                GlobalState.log.warning(f"{len(hidden_states_ids)=} > {len(self.prefix_adapters)=}, this could lead to not using meaningfull features")
            GlobalState.log.info(f"Adapter will use {len(hidden_states_ids)} feature layers from {hidden_states_ids} (out of {len(feat_extractor.hidden_states)} feature layers)")
            GlobalState.log.info(f"Adapter will use {len(self.adapter_connectors)} connectors")

    def forward(self, *args, **kwargs):
        for prefix_adapt_id in range(len(self.prefix_adapters)):
            hidden_state_id = (prefix_adapt_id * len(self.hidden_states)) // len(self.prefix_adapters)
            connector_id = (prefix_adapt_id * len(self.adapter_connectors)) // len(self.prefix_adapters)

            prefix_adapt = self.prefix_adapters[prefix_adapt_id]
            connector = self.adapter_connectors[connector_id]
            collected_feats = self.hidden_states[hidden_state_id].collected_feats
            if self.config.feat_model.hidden_feature_shape != 'auto':
                assert collected_feats[0].shape == self.config.feat_model.hidden_feature_shape, f"Collected features of shape {collected_feats[0].shape} instead of {self.config.feat_model.hidden_feature_shape}"
            if self.config.adapter.merge_layer_features:
                collected_feats = torch.cat([hid.collected_feats for hid in self.hidden_states], dim=-2)

            prefix_adapt.store_extracted_feature(connector(
                collected_feats,
                hidden_state_id,
                prefix_adapt_id,
            ))
        return super().forward(*args, **kwargs)

    def get_llm_module_list(self):
        return unwrap(self).layers

    def build_prefix_adapters(self):
        """ Build wrappers around transformer layers inside the neural network """
        module_list = self.get_llm_module_list()

        # Build the list of layers where a new token will be inserted
        modules_ids = sublist_ids_from_cfg(self.config.adapter.target, n_total=len(module_list))
        GlobalState.log.info(f"Adapter will insert features into {len(modules_ids)} layers {modules_ids} (out of {len(module_list)} llm layers)")
        last_layer_id = (len(module_list)-1) if self.config.adapter.keep_inserted_tokens_until_end else modules_ids[-1]
        flowed_through_modules_ids = sorted(set(range(modules_ids[0], last_layer_id+1)) - set(modules_ids))
        flow_tokens = len(flowed_through_modules_ids) > 0 # If False, then we can insert and delete tokens on same layer
        GlobalState.log.info(f"Tokens will be flowed through {len(flowed_through_modules_ids)} layers {flowed_through_modules_ids} (until end -> {self.config.adapter.keep_inserted_tokens_until_end})")

        prefix_adapters = []
        for mod_id in modules_ids:
            module_list[mod_id] = PrefixTokensOnLayerWithConnector(
                module_list[mod_id],
                insert_at=self.config.adapter.insert_at,
                n_new_tokens=self.n_new_tokens,
                replace_tokens=(mod_id != modules_ids[0]) and flow_tokens,
                remove_tokens_from_output=(mod_id == last_layer_id) or not flow_tokens,
                gating=self.config.adapter.gating,
                token_norm=self.config.adapter.token_norm,
                norm_layer=self.config.adapter.norm_layer,
                padding_mode=self.config.adapter.padding_mode,
                bias_tuning=self.config.adapter.bias_tuning,
            )
            prefix_adapters.append(module_list[mod_id])

        # Build the list of layers where the tokens will will be flowed through
        for mod_id in flowed_through_modules_ids:
            module_list[mod_id] = InsertedTokenPipelinePiece(
                module_list[mod_id],
                insert_at=self.config.adapter.insert_at,
                n_new_tokens=self.n_new_tokens,
                remove_tokens_from_output=(mod_id == last_layer_id),
                padding_mode=self.config.adapter.padding_mode,
            )

        return prefix_adapters

    def _compute_n_new_tokens(self, feats_shape) -> int:
        assert len(feats_shape) == 2
        n_tokens, _ = feats_shape
        cfg = self.config.adapter
        resampler_cfg = self.config.resampler


        if cfg.transformer.n_layers: # Keep only queries if transformer
            n_tokens = cfg.transformer.n_tokens
        else: # Keep only unremoved tokens
            if resampler_cfg.type == 'rand_any_patch':
                self.resampler_n_tokens = VAR_NUM_TOKENS
                return VAR_NUM_TOKENS_LAMBDA(lambda : self.resampler_n_tokens)

            if resampler_cfg.type:
                n_tokens -= 1
                n_tokens //= resampler_cfg.reduction_fact
                if resampler_cfg.type == 'qpmapper':
                    n_tokens *= resampler_cfg.qpmapper.n_tokens
                if not resampler_cfg.drop_cls:
                    n_tokens += resampler_cfg.duplicate_cls

            keep_features_ids = sublist_ids_from_cfg(self.config.adapter.source.keep_features, n_total=n_tokens)
            n_tokens = len(keep_features_ids)
        return n_tokens

    def _build_connector(self, in_shape, out_shape, features_ids, prefixs_ids):
        connector = SequentialConnector(features_ids=features_ids, prefixs_ids=prefixs_ids)
        cfg = self.config.adapter
        insert_activation = False
        assert len(in_shape) == len(out_shape) == 2
        n_vis_tokens, embed_dim = in_shape
        out_n_tokens, out_dim = out_shape

        ##### If not transformer layer, keep only some of the features #####
        if not cfg.transformer.n_layers:
            keep_features_ids = sublist_ids_from_cfg(self.config.adapter.source.keep_features, n_total=in_shape[0])
            if len(keep_features_ids) != in_shape[0]:
                connector.append(DropFeaturesLayer(keep_features_ids))
                n_vis_tokens = len(keep_features_ids)

        ##### If CLS pos encoding, add encoding #####
        if cfg.transformer.pos_encoding == 'CLS':
            connector.append(CLSPosEmbeding())
            embed_dim += 1

        ##### If needed, downsample to the embeding dim #####
        if cfg.embed_dim not in [0, None, 'auto']: # or embed_dim != in_shape[-1]:
            target_dim = cfg.embed_dim
            if target_dim == 'llm_dim':
                target_dim = out_dim
            assert target_dim != "UNK"
            connector.append(torch.nn.Linear(
                embed_dim,
                target_dim,
                bias=cfg.mlp.bias
            ))
            embed_dim = target_dim

        if self.config.resampler.type:
            resampler = TokensResampler(self.config, n_vis_tokens, embed_dim, out_dim)
            n_vis_tokens = resampler.n_out_tokens
            self.resampler_n_tokens = resampler.n_out_tokens
            connector.append(resampler)
            embed_dim = resampler.out_dim

        ##### Insert transformer layer #####
        assert cfg.transformer.pos_encoding in [None, 'CLS', 'rotary', 'learned']
        if cfg.transformer.n_layers:
            if cfg.transformer.mode == "simple":
                # assert cfg.transformer.n_layers == 1
                connector.append(LearnableQueriesAttention(
                    embed_dim=embed_dim,
                    num_heads=cfg.transformer.num_heads,
                    num_queries=cfg.transformer.n_tokens,
                    dropout=cfg.transformer.dropout,
                    bias=cfg.transformer.bias,
                    linear=True,
                    out_dim = (embed_dim if cfg.mlp.n_layers else out_dim),
                ))
            elif cfg.transformer.mode == "QPMapper":
                # assert cfg.mlp.n_layers > 0, "Should have at least one linear layer after transformer"
                connector.append(TransformerFeatureExtractor(
                    embed_dim=embed_dim,
                    num_heads=cfg.transformer.num_heads,
                    num_queries=cfg.transformer.n_tokens,
                    num_layers=cfg.transformer.n_layers,
                    dropout=cfg.transformer.dropout,
                    bias=cfg.transformer.bias,
                    activation=cfg.activation,
                    embed_queries=cfg.transformer.embed_queries,
                    is_causal=cfg.transformer.causal,
                    keep_cls_token = cfg.transformer.keep_cls_token,
                    n_prefix_queries=(len(prefixs_ids) if cfg.transformer.query_per_layer else None),
                    pos_encoding=cfg.transformer.pos_encoding,
                    n_input_tokens=n_vis_tokens,
                ))
            elif cfg.transformer.mode in ["simple_and_Q", "simple_lin_and_Q"]:
                # assert cfg.mlp.n_layers > 0, "Should have at least one linear layer after transformer"
                connector.append(TransformerFeatureExtractor(
                    embed_dim=embed_dim,
                    num_heads=cfg.transformer.num_heads,
                    num_queries=None,
                    num_layers=cfg.transformer.n_layers-1,
                    dropout=cfg.transformer.dropout,
                    bias=cfg.transformer.bias,
                    activation=cfg.activation,
                    embed_queries=False,
                    is_causal=cfg.transformer.causal,
                    keep_cls_token = cfg.transformer.keep_cls_token,
                    n_prefix_queries=(len(prefixs_ids) if cfg.transformer.query_per_layer else None),
                    pos_encoding=cfg.transformer.pos_encoding,
                    n_input_tokens=n_vis_tokens,
                ))
                connector.append(LearnableQueriesAttention(
                    embed_dim=embed_dim,
                    num_heads=cfg.transformer.num_heads,
                    num_queries=cfg.transformer.n_tokens,
                    dropout=cfg.transformer.dropout,
                    bias=cfg.transformer.bias,
                    linear=(cfg.transformer.mode == 'simple_lin_and_Q'),
                    out_dim = (embed_dim if cfg.mlp.n_layers else out_dim),
                ))
            elif cfg.transformer.mode == "Q_former":
                assert cfg.mlp.n_layers > 0, "Should have at least one linear layer after transformer"
                connector.append(QFormerFeatureExtractor(
                    embed_dim=embed_dim,
                    num_heads=cfg.transformer.num_heads,
                    num_queries=cfg.transformer.n_tokens,
                    num_layers=cfg.transformer.n_layers,
                    dropout=cfg.transformer.dropout,
                    bias=cfg.transformer.bias,
                    activation=cfg.activation,
                    n_prefix_queries=(len(prefixs_ids) if cfg.transformer.query_per_layer else None),
                ))
            else:
                raise NotImplementedError(f"No implementation for transformer mode {cfg.transformer.mode}")
            n_vis_tokens = cfg.transformer.n_tokens
            if cfg.transformer.activation_before_mlp:
                insert_activation = True


        ##### Insert linear layer or MLP #####
        if cfg.mlp.n_layers:
            if cfg.mlp.flatten:
                connector.append(torch.nn.Flatten())
            lin_embed_dim = (n_vis_tokens * embed_dim) if cfg.mlp.flatten else embed_dim
            lin_out_dim = np.prod(out_shape) if cfg.mlp.flatten else out_dim

            for i_layer in range(cfg.mlp.n_layers):
                if insert_activation:
                    connector.append(get_activation(cfg.activation))
                insert_activation = True

                if cfg.mlp.dropout:
                    connector.append(torch.nn.Dropout(p=cfg.mlp.dropout))

                connector.append(torch.nn.Linear(
                    lin_embed_dim,
                    lin_embed_dim if i_layer < cfg.mlp.n_layers-1 else lin_out_dim,
                    bias=cfg.mlp.bias
                ))

            if cfg.mlp.flatten:
                connector.append(torch.nn.Unflatten(-1, out_shape))
            embed_dim = out_dim

        if n_vis_tokens != VAR_NUM_TOKENS:
            assert n_vis_tokens == out_n_tokens, f"Got {n_vis_tokens} for the connector, instead of the {out_n_tokens} initially computed"
        assert embed_dim == out_dim

        return connector


class CrossAttentionAdapter(PrefixTokensInnerLayersAdapter):
    def build_prefix_adapters(self):
        """ Build wrappers around transformer layers inside the neural network """
        module_list = self.get_llm_module_list()
        assert self.config.adapter.sharing.share_connections and self.config.adapter.sharing.n_connections == 1, "Sharing not implemented yet"

        # Build the list of layers where a new token will be inserted
        modules_ids = sublist_ids_from_cfg(self.config.adapter.target, n_total=len(module_list))
        GlobalState.log.info(f"Adapter will insert features into {len(modules_ids)} layers {modules_ids} (out of {len(module_list)} llm layers)")

        self.cross_attention = CrossAttentionFeatExtractor(
            llm_dim=get_input_feature_of(module_list[0]),
            embed_dim=self.embed_dim,
            num_heads=self.config.adapter.transformer.num_heads,
            dropout=self.config.adapter.transformer.dropout,
            gating=self.config.adapter.gating,
            norm=self.config.adapter.token_norm,
            out_mlp_layers=self.config.adapter.cross_attn.out_mlp_layers,
            vis_mlp_layers=self.config.adapter.cross_attn.vis_mlp_layers,
            llm_mlp_layers=self.config.adapter.cross_attn.llm_mlp_layers,
            residual=self.config.adapter.cross_attn.residual,
            activation=self.config.adapter.activation,
            norm_layer=self.config.adapter.norm_layer,
            mlp_scale=self.config.adapter.mlp_scale,
        )

        prefix_adapters = []
        for mod_id in modules_ids:
            module_list[mod_id] = CrossAttentionLayerAdapter(
                module_list[mod_id], self.cross_attention, n_vis_tokens=self.n_new_tokens,
            )
            prefix_adapters.append(module_list[mod_id])

        return prefix_adapters


    def _build_connector(self, in_shape, out_shape, features_ids, prefixs_ids):
        out_n_tokens, _ = out_shape
        out_shape = (out_n_tokens, self.embed_dim)
        return super()._build_connector(in_shape, out_shape, features_ids, prefixs_ids)


class PrefixTokensL0Adapter(PrefixTokensInnerLayersAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_dim = get_input_feature_of(self.get_llm_module_list()[0])
        self.connector = self._build_connector(
            in_shape=self.feats_shape,
            out_shape=(self.n_new_tokens, self.prompt_dim),
            features_ids=[0],
            prefixs_ids=[0],
        )
        self._last_n_inserted_tokens = None

    def build_prefix_adapters(self) -> None:
        modules_ids = sublist_ids_from_cfg(self.config.adapter.target, n_total=len(self.get_llm_module_list()))
        assert modules_ids == [0], "Parameters specify injection target different form first layer"
        assert len(self.hidden_states) == 1, "Can't inject from multiple hidden state in layer 0 for now"
        assert self.config.adapter.keep_inserted_tokens_until_end, "Must keep tokens until the end to use this adapter"
        assert self.config.adapter.insert_at == 0

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        assert self.config.llm.model.use_cache, f"Why use this without cache?"
        assert kwargs['use_cache'] is not False, f"Why use this without cache?"
        assert attention_mask is not None
        assert bool(inputs_embeds is None) != bool(input_ids is None)

        features = self.hidden_states[0].collected_feats
        if self.config.feat_model.hidden_feature_shape not in ['auto', 'llm_dim']:
            assert features[0].shape == self.config.feat_model.hidden_feature_shape, f"Collected features of shape {features[0].shape} instead of {self.config.feat_model.hidden_feature_shape}"

        if input_ids is not None:
            inputs_embeds = unwrap(self).embed_tokens(input_ids)

        if kwargs['past_key_values'] is None:
            inserted_tokens = self.connector(features, 0, 0)
            inserted_tokens = inserted_tokens.repeat_interleave(len(inputs_embeds) // len(inserted_tokens), dim=0)
            inputs_embeds = torch.cat([inserted_tokens, inputs_embeds], dim=1)
            self._last_n_inserted_tokens = inserted_tokens.shape[1]

        # Pad attention mask
        attention_mask = torch.nn.functional.pad(attention_mask, (self._last_n_inserted_tokens, 0), value=1)

        kwargs['inputs_embeds'] = inputs_embeds
        kwargs['attention_mask'] = attention_mask
        if 'position_ids' in kwargs:
            kwargs['position_ids'] = None # Will be generated again by LlamaModel.forward

        # out_vals = call_with_splited_batch(self.base_module, kwargs, ref_var='inputs_embeds', max_tokens=self.config.training.batch_max_tokens)
        out_vals = self.base_module(**kwargs) # FORWARD

        # Remove inserted tokens
        if kwargs['past_key_values'] is None:
            out_vals['last_hidden_state'] = out_vals['last_hidden_state'][:, inserted_tokens.shape[1]:]
        return out_vals

class LearnedPrefixTokensL0(ModuleWrapper):
    def __init__(self, base_module, n_new_tokens):
        super().__init__(base_module)

        self.n_new_tokens = n_new_tokens
        layers = unwrap(self).layers
        self.prompt_dim = get_input_feature_of(layers[0])
        if hasattr(unwrap(self), 'project_in') and unwrap(self).project_in is not None:
            self.prompt_dim = unwrap(self).project_in.in_features
        self.prompt_tokens = torch.nn.Parameter(torch.empty((1, self.n_new_tokens, self.prompt_dim)))
        torch.nn.init.normal_(self.prompt_tokens)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        assert attention_mask is not None
        assert bool(inputs_embeds is None) != bool(input_ids is None)

        if input_ids is not None:
            inputs_embeds = unwrap(self).embed_tokens(input_ids)

        if kwargs['past_key_values'] is None:
            inserted_tokens = self.prompt_tokens.repeat((len(inputs_embeds), 1, 1))
            inputs_embeds = torch.cat([inserted_tokens, inputs_embeds], dim=1)

        # Pad attention mask
        attention_mask = torch.nn.functional.pad(attention_mask, (self.n_new_tokens, 0), value=1)

        kwargs['inputs_embeds'] = inputs_embeds
        kwargs['attention_mask'] = attention_mask
        if 'position_ids' in kwargs:
            kwargs['position_ids'] = None # Will be generated again by LlamaModel.forward

        out_vals = self.base_module(**kwargs) # FORWARD

        # Remove inserted tokens
        if kwargs['past_key_values'] is None:
            out_vals['last_hidden_state'] = out_vals['last_hidden_state'][:, inserted_tokens.shape[1]:]
        return out_vals