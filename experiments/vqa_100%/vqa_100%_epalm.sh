#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/vqa_100%/epalm" dataset=vqa llm=opt experiment=epalm exp_name=epalm training.optimizer.args.lr=1e-3
