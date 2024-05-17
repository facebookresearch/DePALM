#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_100%_clap/limber-all_clap" dataset=audiocaps llm=opt training.epochs=12 experiment=limber exp_name=limber-all feat_model=clap dataset.audio_model_type=clap feat_model.load_float16=false dataset.splits.train.batch_size=1 training.accumulate_steps=16
