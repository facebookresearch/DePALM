#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python run.py run_dir="outputs/coco_100%/mapl_eva-clip-L" dataset=coco feat_model=eva-clip-L experiment=mapl exp_name=mapl training.clip_norm=0.2
