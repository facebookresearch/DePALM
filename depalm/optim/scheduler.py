# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .cosine_scheduler import LinearWarmupCosineAnnealingLR

SCHEDULERS = {}

def create_scheduler(accelerator, optimizer, config):
    scheduler_cls = SCHEDULERS[config.training.scheduler.name]
    scheduler_cfg = dict(config.training.scheduler)
    scheduler = scheduler_cls(optimizer, config.training,**scheduler_cfg)

    return scheduler

# ========== Creating the scheduler functions ==========
# The functions under are wrapper around well-knowned schedulers to set defaults and require some arguments

def register_optimizer(scheduler_name):
    def decorator(scheduler_fct):
        SCHEDULERS[scheduler_name.lower()] = scheduler_fct
        return scheduler_fct
    return decorator

@register_optimizer("cosine_no_restarts")
def cosine_scheduler(optimizer, training_cfg, name, scheduler_args):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, training_cfg.epochs, **scheduler_args
    )

@register_optimizer("cosine_warm_restarts")
def cosine_scheduler(optimizer, training_cfg, name, restart_epoch_interval=1, restart_inter_increase_rate=1, min_lr=0.0):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=restart_epoch_interval, T_mult=restart_inter_increase_rate, eta_min=min_lr,
    )

@register_optimizer("cosine")
def cosine_scheduler(optimizer, training_cfg, name, **kwargs):
    return LinearWarmupCosineAnnealingLR(
        optimizer, **kwargs, total_epochs=training_cfg.epochs,
    )