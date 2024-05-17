#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_1%/limber-1_mavil" dataset=audiocaps dataset.splits.train.batch_size=4 training.accumulate_steps=4 llm=opt dataset.splits.train.max_rows=0.01 training.epochs=30 experiment=limber adapter.source.keep_features.n_elements=1 exp_name=limber-1 feat_model=mavil training.optimizer.args.lr=4e-4 training.clip_norm=0.5 feat_model.global_pool=true
