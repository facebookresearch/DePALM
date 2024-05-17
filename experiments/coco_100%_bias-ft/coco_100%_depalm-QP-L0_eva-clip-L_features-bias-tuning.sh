#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/coco_100%/depalm-QP-L0_eva-clip-L_features-bias-tuning" dataset=coco feat_model=eva-clip-L finetune.feat_model.bias_tuning.enable=true dataset.splits.train.batch_size=4 training.accumulate_steps=4 experiment=depalm mod=insert_l0 finetune.prompt_tuning.new_tokens=1 adapter.transformer.n_layers=2 exp_name=depalm-QP-L0 training.optimizer.args.lr=4e-4
