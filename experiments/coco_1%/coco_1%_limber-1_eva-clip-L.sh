#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/coco_1%/limber-1_eva-clip-L" dataset=coco feat_model=eva-clip-L dataset.splits.train.max_rows=0.01 training.epochs=30 experiment=limber adapter.source.keep_features.n_elements=1 exp_name=limber-1 training.optimizer.args.lr=1e-3
