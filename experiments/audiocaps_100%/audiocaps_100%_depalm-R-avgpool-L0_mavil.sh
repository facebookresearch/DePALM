#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/audiocaps_100%/depalm-R-avgpool-L0_mavil" dataset=audiocaps llm=opt training.epochs=12 experiment=depalm_resampler resampler.type=avg_pool adapter.embed_dim=llm_dim exp_name=depalm-R-avgpool-L0 dataset.splits.train.batch_size=8 training.accumulate_steps=2 feat_model=mavil training.clip_norm=0.5 training.optimizer.args.lr=1e-3
