#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/aokvqa_100%/depalm-R-rand-L0" dataset=aokvqa llm=opt training.epochs=20 experiment=depalm_resampler resampler.type=rand_any_patch resampler.reduction_fact=1 adapter.embed_dim=auto training.optimizer.args.lr=1e-3 exp_name=depalm-R-rand-L0 dataset.splits.train.batch_size=8 training.accumulate_steps=2
