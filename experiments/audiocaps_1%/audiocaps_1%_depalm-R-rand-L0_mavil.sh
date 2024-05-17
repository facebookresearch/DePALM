#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_1%/depalm-R-rand-L0_mavil" dataset=audiocaps llm=opt dataset.splits.train.max_rows=0.01 training.epochs=30 experiment=depalm_resampler resampler.type=rand_any_patch resampler.reduction_fact=1 adapter.embed_dim=auto exp_name=depalm-R-rand-L0 feat_model=mavil training.optimizer.args.lr=4e-4 training.clip_norm=0.5 dataset.splits.train.batch_size=8 training.accumulate_steps=2
