#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_100%_clap/limber-1_clap" dataset=audiocaps dataset.splits.train.batch_size=4 training.accumulate_steps=4 llm=opt training.epochs=12 experiment=limber adapter.source.keep_features.n_elements=1 exp_name=limber-1 feat_model=clap dataset.audio_model_type=clap feat_model.load_float16=false training.optimizer.args.lr=1e-3
