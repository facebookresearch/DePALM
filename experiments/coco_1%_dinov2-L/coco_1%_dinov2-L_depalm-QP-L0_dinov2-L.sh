#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/coco_1%_dinov2-L/depalm-QP-L0_dinov2-L" dataset=coco dataset.splits.train.max_rows=0.01 training.epochs=30 feat_model=dinov2-L adapter.transformer.n_tokens=32 experiment=depalm mod=insert_l0 finetune.prompt_tuning.new_tokens=1 adapter.transformer.n_layers=2 exp_name=depalm-QP-L0 training.optimizer.args.lr=2e-3
