# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ...utils.utility import ModuleWrapper, unwrap


class GatedAttentionLayer(ModuleWrapper):
    def __init__(self, layer, from_id, to_id):
        super().__init__(layer)
        # Add parameters to sub-layer
        layer = unwrap(layer)
        assert not hasattr(layer, 'gating_value'), "Layer already used for gating"
        layer.gating_value = torch.nn.Parameter(torch.tensor(0.))
        layer.from_id = from_id
        layer.to_id = to_id

    def forward(self, *args, **kwargs):
        return super().forward(*args, gating=True, **kwargs)