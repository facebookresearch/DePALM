#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/vqa_1%/mapl" dataset=vqa llm=opt dataset.splits.train.max_rows=0.01 training.epochs=30 experiment=mapl exp_name=mapl
