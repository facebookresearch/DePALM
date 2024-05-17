# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
import dataclasses

import torch
import numpy as np
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class GlobalState:
    accelerator = None # IAccelerator
    logger = None
    training_fract = 1

def is_int(value):
    try:
        value = int(value)
        return True
    except:
        return False

class HideStdoutWrites:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class HideStderrWrites:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

def log_mem_info(accelerator):
    n_gpu = torch.cuda.device_count()
    format_mem = lambda mem : f'{mem/1024**3:.2f}'
    for gpu_id in range(n_gpu):
        free, total = torch.cuda.mem_get_info(gpu_id)
        used = total - free
        free, total, used = format_mem(free), format_mem(total), format_mem(used)
        device_name = torch.cuda.get_device_name(device=gpu_id)
        accelerator.log.info(f'Device {gpu_id}: {used} / {total} GB used ({free} free) [{device_name}]')

def var_with_shapes(in_var, stats=True):
    var_stats = ''
    if isinstance(in_var, torch.Tensor):
        if stats:
            var_stats = f' |{in_var.device} [{in_var.min()}-{in_var.float().mean()}-{in_var.max()}]'
        return f'<tensor {tuple(in_var.shape)} {in_var.dtype}{var_stats}>'
    elif isinstance(in_var, np.ndarray):
        if stats:
            var_stats = f' [{in_var.min()}-{in_var.mean()}-{in_var.max()}]'
        return f'<np.ndarray {tuple(in_var.shape)} {in_var.dtype}{var_stats}>'
    elif isinstance(in_var, dict):
        return { key: var_with_shapes(val) for key, val in in_var.items() }
    elif isinstance(in_var, list):
        return [ var_with_shapes(val) for val in in_var ]
    elif isinstance(in_var, tuple):
        return tuple([ var_with_shapes(val) for val in in_var ])
    elif dataclasses.is_dataclass(in_var):
        data_dict = var_with_shapes(dataclasses.asdict(in_var))
        data_dict = ', '.join(f'{key}={val}' for key, val in data_dict.items())
        return f'{in_var.__class__.__name__}({data_dict})'
    else:
        return in_var

def sublist_ids_from_cfg(cfg, n_total):
    """ Generate a sublist of [from_idx ; to_idx], including the two borns """
    from_idx = cfg.from_idx
    to_idx = cfg.to_idx
    stride = 1 if cfg.stride is None else cfg.stride

    n_elements = cfg.n_elements
    if cfg.n_elements == 'all':
        n_elements = n_total
        if from_idx is not None:
            n_elements -= from_idx
        if to_idx is not None:
            n_elements -= (n_total-1 - (to_idx + n_total)%n_total)

    if from_idx is None:
        assert n_elements is not None
        from_idx = to_idx - (n_elements-1) * stride
    if to_idx is None:
        assert n_elements is not None
        to_idx = from_idx + (n_elements-1) * stride

    if not -n_total <= from_idx < n_total:
        raise ValueError(f"Out of bound [0;{n_total}[ for value from_idx={from_idx} in sublist_ids_from_cfg")
    if not -n_total <= to_idx < n_total:
        raise ValueError(f"Out of bound [0;{n_total}[ for value to_idx={to_idx} in sublist_ids_from_cfg")

    from_idx = (from_idx + n_total)%n_total # Including from_idx
    to_idx = (to_idx + n_total)%n_total # Including to_idx
    assert from_idx <= to_idx

    if cfg.n_elements is not None and cfg.stride is not None:
        # Check for incoherent configuration parameters
        assert (to_idx - from_idx) // stride == n_elements-1

    if cfg.stride is not None:
        ids_list = list(range(to_idx, from_idx-1, -stride))[::-1]
    else:
        if n_elements is None:
            n_elements = to_idx - from_idx + 1
        ids_list = list(map(int, np.linspace(from_idx, to_idx, n_elements)))

    assert len(ids_list) <= n_total
    assert len(set(ids_list)) == len(ids_list), "Duplicate ids"

    return ids_list

def merge_dicts(dict_list):
    merged_dict = {}
    for el_dict in dict_list:
        merged_dict.update(el_dict)
    return merged_dict

def str_now():
    return datetime.now().strftime("%y%m%d_%H%M%S")

def get_effective_batch_size(accelerator, config, split='train'):
    return config.dataset.splits[split].batch_size * accelerator.num_processes * config.training.accumulate_steps

def absolute_data_dir_path(path):
    is_path_obj = isinstance(path, Path)
    if path:
        path = Path(path)
        if not path.is_absolute():
            root = Path(__file__).parent.parent.parent
            path = root / path
        path = path.resolve()
        if not is_path_obj:
            path = str(path)
    return path


# ========== Torch utilities ==========

def inner_padding(in_tensor, paddings, value):
    """
        Insert row / columns with a constant value.

        paddings: list / tuple of pairs (insert_at, length). The first is for the last dimension,
            the second for the dimension -2, etc.
    """
    for dim, (insert_at, length) in enumerate(paddings):
        dim = -1-dim
        shape = list(in_tensor.shape)
        shape[dim] = length
        inserting = torch.full(shape, value).to(in_tensor.device)
        in1, in2 = torch.split(in_tensor, [insert_at, in_tensor.shape[dim]-insert_at], dim=dim)
        in_tensor = torch.cat((in1, inserting, in2), dim=dim)
    return in_tensor

class ModuleWrapper(torch.nn.Module):
    def __init__(self, base_module):
        super().__init__()
        self.base_module = base_module

    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)

def unwrap(model):
    if isinstance(model, ModuleWrapper):
        return unwrap(model.base_module)
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return unwrap(model.module)
    elif isinstance(model, torch.distributed.fsdp.FullyShardedDataParallel):
        return unwrap(model.module)
    return model

def uncompiled(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model

class DropFeaturesLayer(torch.nn.Module):
    def __init__(self, keep_features_ids) -> None:
        super().__init__()
        assert isinstance(keep_features_ids, list)
        self.keep_features_ids = keep_features_ids

    def forward(self, tensor):
        return tensor[:, self.keep_features_ids, :] # Along dim=1, because of batch

def get_input_feature_of(layer):
    layer = unwrap(layer)
    if hasattr(layer, 'in_features'): # Linear layer
        return layer.in_features
    elif hasattr(layer, 'embed_dim'): # OPT Attention
        return layer.embed_dim
    elif hasattr(layer, 'hidden_size'): # Llama Attention
        return layer.hidden_size
    else:
        raise NotImplementedError(f"Unsuported layer type: {type(layer)}")

def get_output_features_of(layer):
    if isinstance(layer, torch.nn.Embedding):
        return layer.embedding_dim
    else:
        raise NotImplementedError(f"Unsuported layer type: {type(layer)}")

def load_partial_subdict(model, parameters, strict=False, check_args=True):
    if check_args:
        arg_names = set([name for name, _ in model.named_parameters()])
        for name in parameters:
            if not name in arg_names:
                raise ValueError(f"Argument {name} is not in the model.\nFor the record, here are the parameters: {' '.join(arg_names)}")
    model.load_state_dict(parameters, strict=strict)

def get_activation(name):
    ACTIVATIONS = {
        'relu': torch.nn.ReLU,
        'gelu': torch.nn.GELU,
        'silu': torch.nn.SiLU,
        'sigmoid': torch.nn.Sigmoid,
        'tanh': torch.nn.Tanh,
        'indentity': torch.nn.Identity,
    }
    if name is False or name is None:
        name = 'indentity'
    name = name.lower()
    if name in ACTIVATIONS:
        return ACTIVATIONS[name]()
    else:
        raise ValueError(f"Can't find activation {name}, should be of {', '.join(ACTIVATIONS.keys())}")

def get_norm_layer(name, dim, eps=1e-4):
    LAYERS = {
        'layer_norm': partial(torch.nn.LayerNorm, eps=eps),
        'rms_norm': partial(LlamaRMSNorm, eps=eps),
        'identity': torch.nn.Identity(),
    }
    name = name.lower()
    if name in LAYERS:
        return LAYERS[name](dim)
    else:
        raise ValueError(f"Can't find activation {name}, should be of {', '.join(LAYERS.keys())}")

# ========== Batch spliting ==========
# If there's too much tokens, we make multiple calls for batch parts. Usefull for dynamic number of tokens with large batch sizes

def _merge_outs(out1, out2):
    assert type(out1) == type(out2)
    if isinstance(out1, torch.Tensor):
        return torch.cat([out1, out2], dim=0)
    elif isinstance(out1, dict):
        assert out1.keys() == out2.keys()
        return type(out1)({
            key: _merge_outs(out1[key], out2[key]) for key in out1.keys()
        })
    elif isinstance(out1, (list, tuple)):
        assert len(out1) == len(out2)
        return type(out1)([_merge_outs(o1, o2) for o1, o2 in zip(out1, out2)])
    else:
        assert out1 == out2
        return out1

def call_with_splited_batch(module, kwargs, ref_var, max_tokens):
    batch_size, n_tokens, _ = kwargs[ref_var].shape
    if batch_size <= 1 or n_tokens <= max_tokens:
        print(f"Using {batch_size=} with {n_tokens=}")
        out = module(**kwargs)
        return out

    kwargs_s1, kwargs_s2 = {}, {}
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            assert len(val) == batch_size
            kwargs_s1[key] = val[:batch_size//2]
            kwargs_s2[key] = val[batch_size//2:]
        else:
            kwargs_s1[key] = kwargs_s2[key] = val

    out1 = call_with_splited_batch(module, kwargs_s1, ref_var=ref_var, max_tokens=max_tokens*2)
    out2 = call_with_splited_batch(module, kwargs_s2, ref_var=ref_var, max_tokens=max_tokens*2)
    return _merge_outs(out1, out2)
