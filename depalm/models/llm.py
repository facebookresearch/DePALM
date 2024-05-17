# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from ..utils.utility import ModuleWrapper, unwrap, GlobalState


def mem_args(config):
    return {
        'low_cpu_mem_usage': False,
    }

def get_model_path(model_name, cache_dir):
    model_name_dir = 'models--' + model_name.replace('/', '--')
    model_path = Path(cache_dir) / model_name_dir / 'snapshots'
    model_path = list(model_path.iterdir())[0]
    return model_path

def autoload_tokenizer(llm_cfg, tokenizerCls=AutoTokenizer):
    model_path = get_model_path(llm_cfg.tokenizer.name, llm_cfg.cache_dir)
    tokenizer = tokenizerCls.from_pretrained(model_path, use_fast=False,
        cache_dir=llm_cfg.cache_dir, local_files_only=llm_cfg.local_files_only,
        legacy=False,
    )

    if llm_cfg.tokenizer.padding_side is not None:
        tokenizer.padding_side = llm_cfg.tokenizer.padding_side
    if llm_cfg.tokenizer.pad_token is not None:
        if isinstance(llm_cfg.tokenizer.pad_token, int):
            tokenizer.pad_token = tokenizer.pad_token
        elif llm_cfg.tokenizer.pad_token == '<eos>':
            tokenizer.pad_token = tokenizer.eos_token
        elif llm_cfg.tokenizer.pad_token == '<bos>':
            tokenizer.pad_token = tokenizer.bos_token
        elif llm_cfg.tokenizer.pad_token == '<unk>':
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError(f"Can't interpret padding with token {llm_cfg.tokenizer.pad_token}")
    if llm_cfg.special_answer_token is not None:
        tokenizer.add_special_tokens({'additional_special_tokens': [llm_cfg.special_answer_token]})
        GlobalState.log.info(f"Adding special token: {llm_cfg.special_answer_token}")
        GlobalState.log.info(tokenizer)

    return tokenizer

def autoload_model_text(llm_cfg, config, wrapper_cls, configCls=AutoConfig, modelCls=AutoModelForCausalLM):
    model_path = get_model_path(llm_cfg.tokenizer.name, llm_cfg.cache_dir)
    model_config = configCls.from_pretrained(model_path,
        cache_dir=llm_cfg.cache_dir, local_files_only=llm_cfg.local_files_only)

    if llm_cfg.model.use_cache is not None:
        model_config.use_cache = llm_cfg.model.use_cache
    assert model_config.use_cache is False, "Can't use cache for generation"

    load_kwargs = {}
    if config.llm.load_float16:
        load_kwargs['torch_dtype'] = torch.float16
    model = modelCls.from_pretrained(model_path, config=model_config, **mem_args(config),
        cache_dir=llm_cfg.cache_dir, local_files_only=llm_cfg.local_files_only,
        **load_kwargs
    )
    model._using_config = model_config

    return wrapper_cls(model)


# ========== LLM Wrappers ==========

class LLMWrapperModel(ModuleWrapper):
    features = None
    def __init__(self, base_module):
        super().__init__(base_module)

    def forward(self, *args, **kwargs):
        return unwrap(self.base_module)(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return unwrap(self.base_module).generate(*args, **kwargs)

    @property
    def word_embeding_dim(self) -> int:
        raise NotImplementedError

    @property
    def decoder(self):
        raise NotImplementedError

    def set_decoder(self, new_decoder):
        raise NotImplementedError

    def wrap_blocks(self, wrapper_fun):
        raise NotImplementedError


class OptWrapper(LLMWrapperModel):
    @property
    def word_embeding_dim(self) -> int:
        return unwrap(self)._using_config.word_embed_proj_dim

    @property
    def decoder(self):
        return unwrap(unwrap(self).model).decoder

    def set_decoder(self, new_decoder):
        unwrap(unwrap(self).model).decoder = new_decoder

    def wrap_blocks(self, wrapper_fun):
        un_decoder = unwrap(self.decoder)
        for id_l, layer in enumerate(un_decoder.layers):
            un_decoder.layers[id_l] = wrapper_fun(id_l, len(un_decoder.layers), layer)


class LlamaWrapper(LLMWrapperModel):
    @property
    def word_embeding_dim(self) -> int:
        return unwrap(self)._using_config.hidden_size

    @property
    def decoder(self):
        return unwrap(self).model

    def set_decoder(self, new_decoder):
        unwrap(self).model = new_decoder

    def wrap_blocks(self, wrapper_fun):
        un_decoder = unwrap(self.decoder)
        for id_l, layer in enumerate(un_decoder.layers):
            un_decoder.layers[id_l] = wrapper_fun(id_l, len(un_decoder.layers), layer)
