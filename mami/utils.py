# SPDX-License-Identifier: MIT
# Adapted from MST-plus-plus (caiyuanhao1998/MST-plus-plus), licensed under the MIT License.
# Copyright (c) <Yuanhao Cai>.
# Modifications Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()
    
    def forward(self, outputs, label) -> Tensor:
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        return torch.mean(error.reshape(-1))

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label) -> Tensor:
        assert outputs.shape == label.shape
        error = outputs - label
        sqrt_error = torch.pow(error, 2)
        return torch.sqrt(torch.mean(sqrt_error.reshape(-1)))

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255) -> Tensor:
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]

        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        
        mse = nn.MSELoss(reduction="none")
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        
        return torch.mean(psnr)
    
class VegetationIndexLoss(nn.Module):
    """
    Generic vegetation-index loss:
        VI = (A - B) / (A + B + eps)

    Computes L1 distance between predicted and target VI maps.

    Examples:
      NDVI: A=NIR, B=RED
      NDRE: A=NIR, B=RED-EDGE
    """
    def __init__(self, a_idx: int, b_idx: int, eps: float = 1e-6):
        super().__init__()
        self.a_idx = int(a_idx)
        self.b_idx = int(b_idx)
        self.eps = float(eps)

    def _index(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected tensor [B, C, H, W], got shape {tuple(x.shape)}")
        c = x.shape[1]
        if not (0 <= self.a_idx < c and 0 <= self.b_idx < c):
            raise ValueError(
                f"Band index out of range for tensor with C={c}: a_idx={self.a_idx}, b_idx={self.b_idx}"
            )

        a = x[:, self.a_idx:self.a_idx + 1, ...]
        b = x[:, self.b_idx:self.b_idx + 1, ...]

        a = a.float()
        b = b.float()

        denom = a + b

        denom = torch.where(
            denom >= 0,
            torch.clamp(denom, min=1e-4),
            torch.clamp(denom, max=-1e-4)
        )

        vi = (a - b) / denom
        return vi

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_vi = self._index(pred)
        tgt_vi = self._index(target)
        return torch.mean(torch.abs(pred_vi - tgt_vi))


class Loss_NDVI(VegetationIndexLoss):
    def __init__(self, nir_idx: int, red_idx: int, eps: float = 1e-6):
        super().__init__(a_idx=nir_idx, b_idx=red_idx, eps=eps)


class Loss_NDRE(VegetationIndexLoss):
    def __init__(self, nir_idx: int, rededge_idx: int, eps: float = 1e-6):
        super().__init__(a_idx=nir_idx, b_idx=rededge_idx, eps=eps)
