#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/vqa_1%_dinov2-L/depalm-QP-inner_dinov2-L" dataset=vqa llm=opt dataset.splits.train.max_rows=0.01 training.epochs=30 feat_model=dinov2-L adapter.transformer.n_tokens=32 experiment=depalm adapter.transformer.n_layers=2 exp_name=depalm-QP-inner dataset.splits.train.batch_size=8 training.accumulate_steps=2 training.optimizer.args.lr=1e-3
