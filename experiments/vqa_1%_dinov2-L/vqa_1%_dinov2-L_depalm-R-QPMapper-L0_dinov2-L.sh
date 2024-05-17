#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/vqa_1%_dinov2-L/depalm-R-QPMapper-L0_dinov2-L" dataset=vqa llm=opt dataset.splits.train.max_rows=0.01 training.epochs=30 feat_model=dinov2-L adapter.transformer.n_tokens=32 experiment=depalm_resampler resampler.type=qsformer adapter.embed_dim=768 training.clip_norm=0.5 exp_name=depalm-R-QPMapper-L0 dataset.splits.train.batch_size=8 training.accumulate_steps=2 adapter.transformer.n_layers=1 training.optimizer.args.lr=4e-4
