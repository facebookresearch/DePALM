#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/coco_100%_dinov2-L/depalm-c-attn_dinov2-L" dataset=coco feat_model=dinov2-L adapter.transformer.n_tokens=32 experiment=depalm mod=cross_attn exp_name=depalm-c-attn dataset.splits.train.batch_size=8 training.accumulate_steps=2
