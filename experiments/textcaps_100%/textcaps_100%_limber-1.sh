#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/textcaps_100%/limber-1" dataset=textcaps llm=llama training.epochs=20 experiment=limber adapter.source.keep_features.n_elements=1 exp_name=limber-1
