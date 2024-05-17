#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/msrvtt_100%/depalm-QP-L0_timesformer" dataset=msrvtt llm=llama experiment=depalm mod=insert_l0 finetune.prompt_tuning.new_tokens=1 adapter.transformer.n_layers=2 exp_name=depalm-QP-L0 feat_model=timesformer dataset.splits.train.batch_size=8 training.accumulate_steps=2 training.optimizer.args.lr=1e-3
