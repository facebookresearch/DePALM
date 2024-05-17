#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/msrvtt_100%/depalm-R-rand-L0_timesformer" dataset=msrvtt llm=llama experiment=depalm_resampler resampler.type=rand_any_patch resampler.reduction_fact=1 adapter.embed_dim=auto exp_name=depalm-R-rand-L0 feat_model=timesformer feat_model.average_time=true dataset.splits.train.batch_size=2 training.accumulate_steps=8 training.optimizer.args.lr=4e-4
