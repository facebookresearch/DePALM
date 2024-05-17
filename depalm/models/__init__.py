# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from .llm import autoload_tokenizer, autoload_model_text, OptWrapper, LlamaWrapper
from .perceptual.vision import autoload_timm_model, autoload_openclip_model, autoload_transformers_model, load_dinov2, load_ast_model, load_dummy_disabled_model
from .perceptual.mavil import load_mavil_base_model
from .perceptual.audio import load_clap_model, CLAPExtractor
from .perceptual.video import load_timesformer_model, TimeSFormerExtractor
from .extractors import VisVitExtractor, DisabledExtractor, DinoV2Extractor, NFNetExtractor, ASTExtractor, MavilExtractor
from .adapters.prefix_tokens import PrefixTokensInnerLayersAdapter, CrossAttentionAdapter, PrefixTokensL0Adapter
from .adapters.common import apply_adapter_transformers


TOKENIZER_LOADERS = {
    'opt': autoload_tokenizer,
    'llama': partial(autoload_tokenizer, tokenizerCls=LlamaTokenizer),
    'vicuna': autoload_tokenizer,
}

LLM_LOADERS = {
    'opt': partial(autoload_model_text, wrapper_cls=OptWrapper),
    'llama': partial(autoload_model_text, wrapper_cls=LlamaWrapper, configCls=LlamaConfig, modelCls=LlamaForCausalLM),
    'vicuna': partial(autoload_model_text, wrapper_cls=LlamaWrapper),
}

ADAPTATER_LOADERS = {
    '*': {
        'prefix_tokens_inner_layers': apply_adapter_transformers(PrefixTokensInnerLayersAdapter),
        'prefix_tokens_l0': apply_adapter_transformers(PrefixTokensL0Adapter),
        'cross_attention': apply_adapter_transformers(CrossAttentionAdapter),
    }
}

FEAT_LOADERS = {
    'timm_vit': autoload_timm_model,
    'openclip': autoload_openclip_model,
    'transformers': autoload_transformers_model,
    'dinov2': load_dinov2,
    'ast': load_ast_model,
    'mavil': load_mavil_base_model,
    'clap': load_clap_model,
    'timesformer': load_timesformer_model,
    'disabled': load_dummy_disabled_model,
}

EXTRACTOR_LOADERS = {
    'timm_vit': VisVitExtractor,
    'openclip': VisVitExtractor,
    'transformers': VisVitExtractor,
    'dinov2': DinoV2Extractor,
    'ast': ASTExtractor,
    'mavil': MavilExtractor,
    'clap': CLAPExtractor,
    'timesformer': TimeSFormerExtractor,
    'disabled': DisabledExtractor,
}
