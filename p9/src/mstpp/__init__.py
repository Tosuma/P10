# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import torch
from torch import nn
from .mstpp import MST_Plus_Plus

def load_modified_mst_pp(model_path: str, device: torch.device) -> nn.Module:
    model = MST_Plus_Plus().cuda()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
            
        if key in model_sd and v.size() == model_sd[key].size():
            filtered[key] = v

    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model