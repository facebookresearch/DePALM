#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/okvqa_100%/depalm-R-QPMapper-L0" dataset=okvqa llm=opt training.epochs=20 experiment=depalm_resampler resampler.type=qsformer adapter.embed_dim=768 training.optimizer.args.lr=1e-3 training.clip_norm=0.5 exp_name=depalm-R-QPMapper-L0 dataset.splits.train.batch_size=8 training.accumulate_steps=2
