# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from ..utils.utility import get_effective_batch_size, unwrap, GlobalState
from ..models.utils import get_parameter_category, PARAMS_CATEGORIES


OPTIMIZERS = {}

def extract_trainable_parameters(depalm_model, optim_cfg=None):
    parameters = {name: param for name, param in depalm_model.named_parameters() if param.requires_grad}
    params_by_cat = {cat: {} for cat in PARAMS_CATEGORIES}

    for name, torch_param in parameters.items():
        params_by_cat[get_parameter_category(name)][name] = torch_param

    GlobalState.log.info('Trainable parameters by category:')
    for cat, cat_params in params_by_cat.items():
        size_params = sum([p.numel() for p in cat_params.values()])
        if size_params:
            overwrides = optim_cfg.overwrite[cat] if optim_cfg is not None else '?'
            GlobalState.log.info(f'- {cat}: {size_params:,} trainable parameters (accross {len(cat_params)} tensors) [overwrides: {overwrides}]')
            if cat == 'other':
                GlobalState.log.warning('Parameter without a category!')
        else:
            GlobalState.log.info(f'- {cat}: no parameters')
    return params_by_cat

def create_optimizer(depalm_model, optim_cfg):
    params_by_cat = extract_trainable_parameters(depalm_model, optim_cfg)
    unw_model = unwrap(depalm_model)
    parameters = [
        {
            'params': [torch_param for torch_param in cat_params.values()],
            'name': cat,
            **optim_cfg.overwrite[cat],
        }
        for cat, cat_params in params_by_cat.items() if cat_params
    ]

    opt_cls = OPTIMIZERS[optim_cfg.name.lower()]
    optimizer = opt_cls(parameters, **optim_cfg.args)
    return optimizer

# ========== Creating the optimizer functions ==========
# The functions under are wrapper around well-knowned optimizers to set defaults and require some arguments

def register_optimizer(optim_name):
    def decorator(optim_fct):
        OPTIMIZERS[optim_name.lower()] = optim_fct
        return optim_fct
    return decorator

register_optimizer("SGD")(torch.optim.SGD)
register_optimizer("Adam")(torch.optim.Adam)
register_optimizer("AdamW")(torch.optim.AdamW)
