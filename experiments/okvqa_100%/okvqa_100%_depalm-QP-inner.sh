#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/okvqa_100%/depalm-QP-inner" dataset=okvqa llm=opt training.epochs=20 experiment=depalm adapter.transformer.n_layers=2 exp_name=depalm-QP-inner training.optimizer.args.lr=1e-3
