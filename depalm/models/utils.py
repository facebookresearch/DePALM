# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import re


PARAMS_CATEGORIES = ['adapter', 'feat_model', 'model_text', 'prompt', 'cross_attn', 'other']

def get_parameter_category(name):
    param_path = name
    cat = 'other'

    if 'prompt_tokens' in param_path:
        cat = 'prompt'
    elif 'cross_attention' in param_path:
        cat = 'cross_attn'
    elif 'adapter_' in param_path or 'gating_value' in param_path:
        cat = 'adapter'
    elif 'model_text' in param_path:
        cat = 'model_text'
    elif 'feat_model' in param_path:
        cat = 'feat_model'
    return cat

def set_submodule(model, name, submodule, strict=True):
    if '.' in name:
        sub_name, name = name.rsplit('.', 1)
        model = model.get_submodule(sub_name)
    if strict:
        assert getattr(model, name) is not None
    setattr(model, name, submodule)

def find_submodule_with_name(model, name_end, strict=True):
    for mod_name, module in model.named_modules():
        if mod_name.endswith(name_end):
            return module
    else:
        if strict:
            raise Exception(f"No submodule matching '{name_end}'")

def find_parameter_with_name(model, name_end, strict=True):
    for par_name, parameter in model.named_parameters():
        if par_name.endswith(name_end):
            return parameter
            break
    else:
        if strict:
            raise Exception(f"No parameter matching '{name_end}'")


def freeze_whole_model(model, freeze=True):
    for _, param in model.named_parameters():
        param.requires_grad = (not freeze)

def unfreeze_parameters_of_layers_types(model, *layer_types, as_float32=False):
    for layer_name, layer in model.named_modules():
        if any([isinstance(layer, lt) for lt in layer_types]):
            for p_name, param in layer.named_parameters():
                param.requires_grad = True
            if as_float32:
                set_submodule(model, layer_name, layer.float())

def unfreeze_parameters(model, parameters):
    parameters = set([p.replace('.base_module', '') for p in parameters])
    for name, param in model.named_parameters():
        name = name.replace('.base_module', '')
        if name in parameters:
            param.requires_grad = True
            parameters.discard(name)
    if parameters:
        raise ValueError(f"Can't find parameters {list(parameters)}")

def unfreeze_parameters_regex(model, config):
    raise NotImplementedError


def show_trainable_params_percent(logger, model, model_name):
    orig_param_size = sum(p.numel() for p in model.parameters())
    trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

    percentage = trainable_size / orig_param_size * 100
    logger.info(f"Trainable param percentage of {model_name}: {percentage:.5f}% ({trainable_size:,}/{orig_param_size:,})")

    # Show parameters by group
    params_groups = defaultdict(lambda : defaultdict(lambda: 0))
    for path_name, param in model.named_parameters():
        if param.requires_grad and param.numel() > 0:
            name = path_name
            name = re.sub(r"\.[0-9]+", "", name) # Remove numbers
            name = re.sub(r"\.bias$", "", name)
            name = re.sub(r"\.weight$", "", name)
            # Remove or rename some layer names
            name = re.sub(r"\.linear[0-9]*$", "", name)
            name = re.sub(r"\.norm[0-9]*$", ".norm", name)
            name = re.sub(r"adapter_connectors\.blocks.*$", "adapter_connectors.transformer_blocks", name)
            name = re.sub(r"\.adapter_bias_tuning\..*$", ".adapter_bias_tuning", name)

            name = name.replace('_fsdp_wrapped_module.', '')
            name = name.replace('.base_module', '')

            if '.bias_tuner' in name:
                if 'model_text' in name:
                    name = 'model_text.bias_tunner'
                elif 'feat_model' in name:
                    name = 'feat_model.bias_tunner'
            elif 'norm' in name:
                pass
            name = name.replace('model_text.model.', 'model_text.')
            name = name.replace('model_text.decoder', 'model_text').replace('model_text.layers', 'model_text')

            params_groups[get_parameter_category(path_name)][name] += param.numel()

    for category, cat_params in params_groups.items():
        for name, numel in cat_params.items():
            percentage = numel / trainable_size * 100
            logger.info(f'{name} [{category}]: {numel:,} parameters ({percentage:.2f}% of trainable)')
    return percentage