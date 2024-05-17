#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/coco_100%/depalm-R-avgpool-L0_eva-clip-L_features-bias-tuning" dataset=coco feat_model=eva-clip-L finetune.feat_model.bias_tuning.enable=true experiment=depalm_resampler resampler.type=avg_pool adapter.embed_dim=llm_dim exp_name=depalm-R-avgpool-L0 dataset.splits.train.batch_size=8 training.accumulate_steps=2 training.optimizer.args.lr=4e-4
