#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/msrvtt_100%/limber-1_timesformer" dataset=msrvtt llm=llama experiment=limber adapter.source.keep_features.n_elements=1 exp_name=limber-1 feat_model=timesformer dataset.splits.train.batch_size=2 training.accumulate_steps=8 training.optimizer.args.lr=1e-3
