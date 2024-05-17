#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_1%/depalm-QP-inner_mavil" dataset=audiocaps dataset.splits.train.batch_size=4 training.accumulate_steps=4 llm=opt dataset.splits.train.max_rows=0.01 training.epochs=30 experiment=depalm adapter.transformer.n_layers=2 exp_name=depalm-QP-inner feat_model=mavil training.optimizer.args.lr=4e-4 training.clip_norm=0.5
