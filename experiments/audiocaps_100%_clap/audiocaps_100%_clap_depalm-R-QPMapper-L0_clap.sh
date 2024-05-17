#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_100%_clap/depalm-R-QPMapper-L0_clap" dataset=audiocaps llm=opt training.epochs=12 experiment=depalm_resampler resampler.type=qsformer adapter.embed_dim=768 training.clip_norm=0.5 exp_name=depalm-R-QPMapper-L0 feat_model=clap dataset.audio_model_type=clap feat_model.load_float16=false dataset.splits.train.batch_size=4 training.accumulate_steps=4 training.optimizer.args.lr=4e-4
