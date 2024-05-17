#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/msrvtt_100%/mapl_timesformer" dataset=msrvtt llm=llama experiment=mapl exp_name=mapl feat_model=timesformer dataset.splits.train.batch_size=8 training.accumulate_steps=2
