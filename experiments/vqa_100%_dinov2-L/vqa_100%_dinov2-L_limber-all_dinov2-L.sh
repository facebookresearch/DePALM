#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/vqa_100%_dinov2-L/limber-all_dinov2-L" dataset=vqa llm=opt feat_model=dinov2-L adapter.transformer.n_tokens=32 experiment=limber exp_name=limber-all dataset.splits.train.batch_size=4 training.accumulate_steps=4
