# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Some code of this file comes from https://github.com/huggingface/accelerate (see https://github.com/huggingface/accelerate/blob/main/LICENSE)
import os
import random
import logging
import datetime
import pickle
from functools import partial
from contextlib import contextmanager, suppress

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, BackwardPrefetch
from torch.cuda.amp import GradScaler

from .utility import unwrap, GlobalState


def get_fsdp_wrapped_modules(model):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    from ..depalmodel import Depalm
    from ..models.extractors import FeatExtractLayerWrapper, FeatExtractorWrapper
    from ..models.adapters.prefix_tokens import ABCPrefixTokensOnLayer

    wrap_cls = {
        torch.nn.MultiheadAttention,
        OPTDecoderLayer,
        LlamaDecoderLayer,
        # FeatExtractLayerWrapper,
        ABCPrefixTokensOnLayer,
    }

    if isinstance(model, Depalm):
        wrap_cls.add(unwrap(model.feat_model.get_blocks()[0]).__class__)

    return wrap_cls

def module_wrap_policy(
    module,
    recurse: bool,
    nonwrapped_numel: int,
    module_classes,
) -> bool:
    if is_compiled_module(module):
        return False # Always return no for compiled module (they have their FSDP inside)
    if recurse:
        return True # Always recurse
    return isinstance(module, tuple(module_classes))


class DummyGradScaler:
    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        pass

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass


class IAccelerator:
    TIMEOUT_NCCL_MIN = 10
    MAX_SHARE_BYTES = 10**7 # 10M

    def __init__(self, local_rank, world_size, dis_cfg):
        self.local_rank = local_rank
        self.world_size = world_size
        self.dis_cfg = dis_cfg
        if world_size == 0:
            self.device = torch.device('cpu')
            dis_cfg.enable = False
            dis_cfg.float_type = "float32"
            dis_cfg.cast_float16 = False
        else:
            self.device = torch.device(self.local_rank)
        self.dtype = self.get_float_type(dis_cfg.float_type)
        self.input_dtype = self.dtype if self.dis_cfg.cast_float16 else torch.float32

        self.dis_enabled = self.dis_cfg.enable
        self.uses_fsdp = self.dis_cfg.fsdp and self.dis_enabled
        self.manage_mp = not self.uses_fsdp or self.dis_cfg.cast_float16 # FSDP manages MP on its own
        self.scaler = GradScaler() if self.manage_mp and not self.dis_cfg.cast_float16 else DummyGradScaler()

        IAccelerator.log = self.get_logger("depalm")
        GlobalState.log = self.get_logger("depalm")
        GlobalState.accelerator = self

        if world_size > 0:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(dis_cfg.master_port or '29345')
            dist.init_process_group("nccl", rank=local_rank, world_size=world_size, timeout=datetime.timedelta(seconds=int(60 * self.TIMEOUT_NCCL_MIN)))
            torch.cuda.set_device(self.device)

    def __repr__(self):
        return f'<IAccelerator local_rank={self.local_rank}/{self.world_size}, device={self.device}>'

    def set_seed(self, seed: int, device_specific: bool = True):
        if device_specific:
            seed += self.local_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @staticmethod
    def get_float_type(float_type):
        if float_type in [None, "float32"]:
            return torch.float32
        elif float_type == "float16":
            return torch.float16
        else:
            raise ValueError(f"Don't know float type {float_type}")

    def autocast(self):
        if self.manage_mp:
            return torch.autocast(device_type='cuda', dtype=self.dtype)
        else:
            return suppress()

    def get_logger(self, name: str, log_level = logging.INFO):
        logger = logging.getLogger(name)
        if log_level is not None:
            logger.setLevel(log_level)
        return MultiProcessAdapter(logger, {}, main_process=self.is_main_process)

    def prepare_model(self, model, compile=False, auto_wrap=True):
        if not self.dis_enabled:
            return model

        if self.dis_cfg.cast_float16:
            assert self.dtype == torch.float16
            model = model.half()

        if self.uses_fsdp:
            # Get the list of parameters with grad + unfreeze model
            grad_on = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    grad_on.append(name)
                param.requires_grad = True

            # FSDP
            auto_wrap_policy = partial(module_wrap_policy, module_classes=get_fsdp_wrapped_modules(model))
            model = FullyShardedDataParallel(
                model,
                device_id = self.local_rank,
                use_orig_params = True,
                auto_wrap_policy = auto_wrap_policy if auto_wrap else None,
                mixed_precision = MixedPrecision(
                    param_dtype = self.dtype,
                    reduce_dtype = self.dtype, # Gradient communication precision.
                    buffer_dtype = self.dtype, # Buffer precision.
                ) if not self.dis_cfg.cast_float16 else None,
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE, # Trade some memory for increased speed
            )

            # Freeze again model
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze the parameters we want to learn
            for name in grad_on:
                model.get_parameter(name).requires_grad = True
        else:
            model = model.to(self.device)
            model = DistributedDataParallel(model, device_ids=[self.local_rank], find_unused_parameters=True)

        if compile:
            mode = None # "default", "reduce-overhead", "max-autotune"
            model = torch.compile(model, mode=mode, dynamic=True)
            assert is_compiled_module(model)

        return model

    def backward(self, loss, **kwargs):
        self.scaler.scale(loss).backward(**kwargs)

    def step(self, model, optimizer, zero_grad=True, clip_grad=None):
        self.scaler.unscale_(optimizer)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=False)
        self.scaler.step(optimizer)
        self.scaler.update()
        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

    def unwrap_model(self, model):
        # From https://github.com/huggingface/accelerate/blob/87c81315a1b71da5d6a9129c9d2dc9a31c794bb6/src/accelerate/utils/other.py#L44
        options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

        is_compiled = is_compiled_module(model)
        if is_compiled:
            compiled_model = model
            model = model._orig_mod

        while isinstance(model, options):
            model = model.module

        if is_compiled:
            compiled_model._orig_mod = model
            model = compiled_model

        return model

    ##### Properties #####

    @property
    def is_main_process(self):
        return self.local_rank == 0

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def num_workers(self):
        if self.dis_enabled:
            return self.dis_cfg.cpu_per_process
        else:
            return 1

    @property
    def num_processes(self):
        return self.world_size

    ##### Synchronization & communication #####

    def barrier(self):
        if self.dis_enabled:
            torch.distributed.barrier()

    @contextmanager
    def main_process_first(self):
        if not self.is_main_process:
            self.barrier()
        yield
        if self.is_main_process:
            self.barrier()

    def all_gather_any(self, obj):
        if self.dis_enabled:
            obj = pickle.dumps(obj)

            # Share objects lengths
            len_list = [None] * self.num_processes
            torch.distributed.all_gather_object(len_list, len(obj))
            max_len = max(len_list)

            all_obj_list = [[] for _ in range(self.num_processes)]
            for i_split in range(0, max_len, self.MAX_SHARE_BYTES):
                obj_list = [None] * self.num_processes
                torch.distributed.all_gather_object(obj_list, obj[i_split:i_split+self.MAX_SHARE_BYTES])
                for proc_obj, proc_list in zip(obj_list, all_obj_list):
                    proc_list.append(proc_obj)

            all_obj_list = [pickle.loads(b''.join(seq)) for seq in all_obj_list]

            return all_obj_list
        else:
            return [obj]

    def all_gather(self, tensor):
        if self.dis_enabled:
            tensor_list = [None] * self.num_processes
            torch.distributed.all_gather_object(tensor_list, tensor)
            return tensor_list
        else:
            return [tensor]

class MultiProcessAdapter(logging.LoggerAdapter):
    def __init__(self, *args, main_process, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_process = main_process

    def log(self, level, msg, *args, main_process_only=True, **kwargs):
        if self.isEnabledFor(level):
            if self.main_process or not main_process_only:
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)

def is_compiled_module(module):
    if not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)