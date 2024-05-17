#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/vqa_100%/depalm-R-linear-L0" dataset=vqa llm=opt experiment=depalm_resampler resampler.type=linear adapter.embed_dim=llm_dim exp_name=depalm-R-linear-L0 dataset.splits.train.batch_size=8 training.accumulate_steps=2 training.optimizer.args.lr=4e-4
