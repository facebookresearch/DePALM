#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_100%/depalm-c-attn_mavil" dataset=audiocaps dataset.splits.train.batch_size=4 training.accumulate_steps=4 llm=opt training.epochs=12 experiment=depalm mod=cross_attn exp_name=depalm-c-attn feat_model=mavil training.optimizer.args.lr=4e-4 training.clip_norm=0.5
