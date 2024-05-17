#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_100%/depalm-R-rand-L0_mavil" dataset=audiocaps llm=opt training.epochs=12 experiment=depalm_resampler resampler.type=rand_any_patch resampler.reduction_fact=1 adapter.embed_dim=auto exp_name=depalm-R-rand-L0 dataset.splits.train.batch_size=8 training.accumulate_steps=2 feat_model=mavil training.clip_norm=0.5 training.optimizer.args.lr=1e-3
