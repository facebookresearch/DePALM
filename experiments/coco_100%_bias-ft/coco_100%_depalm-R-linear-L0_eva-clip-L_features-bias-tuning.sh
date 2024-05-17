#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/coco_100%/depalm-R-linear-L0_eva-clip-L_features-bias-tuning" dataset=coco feat_model=eva-clip-L finetune.feat_model.bias_tuning.enable=true experiment=depalm_resampler resampler.type=linear adapter.embed_dim=llm_dim exp_name=depalm-R-linear-L0 dataset.splits.train.batch_size=2 training.accumulate_steps=8 training.optimizer.args.lr=4e-4
