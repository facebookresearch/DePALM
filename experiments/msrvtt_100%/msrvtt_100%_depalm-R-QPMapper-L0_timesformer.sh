#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/msrvtt_100%/depalm-R-QPMapper-L0_timesformer" dataset=msrvtt llm=llama experiment=depalm_resampler resampler.type=qsformer adapter.embed_dim=768 training.clip_norm=0.5 exp_name=depalm-R-QPMapper-L0 feat_model=timesformer dataset.splits.train.batch_size=1 training.accumulate_steps=16 training.optimizer.args.lr=4e-4
