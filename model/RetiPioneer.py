from typing import List

import torch
import torch.nn as nn

from model.base import get_RETFound, get_SwinB, get_VimS
from model.QualityAware import QualityAware


class FuseBase(nn.Module):
    def __init__(self,
                 base: nn.Module,
                 base_out_size: int,
                 meta_size: int,
                 num_classes: int,
                 fuse_dim: int,
                 enable_q: bool = True,
                 learnable_q: bool = False,
                 flip_r: bool = True) -> None:
        super().__init__()

        self.base = base
        self.fuse_dim = fuse_dim
        self.quality_aware = QualityAware(base_out_size, self.fuse_dim, enable_q, learnable_q)
        self.m_fuse = nn.Bilinear(self.fuse_dim * 2 + 1, meta_size + 1, num_classes, False)
        self.flip_r = flip_r

    def forward(self, xmq: List[torch.Tensor]) -> torch.Tensor:
        lr, m, qs = xmq
        dev = m.device
        bat = m.shape[0]

        if self.flip_r:
            lr = [lr[0], lr[1].flip([-1])]
        else:
            lr = [lr[0], lr[1]]

        xqf2 = [torch.ones(bat, 1, device=dev)]
        for x, q in zip(lr, qs):
            xf = self.base(x)
            xqf2.append(self.quality_aware(xf, q))
        xqf2 = torch.cat(xqf2, dim=1)
        m = torch.cat([torch.ones(bat, 1, device=dev), m], dim=1)
        xqmf = self.m_fuse(xqf2, m)

        return xqmf


class ComplexModel(nn.Module):
    def __init__(self, backbones: List[nn.Module], base_out_sizes: List[int], mid_size: int, fuse_dim: int, n_meta: int, num_classes: int):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                FuseBase(backbone, size, n_meta, mid_size, fuse_dim, flip_r=False),
                nn.SELU(True),
                nn.Linear(mid_size, num_classes)
            ) for backbone, size in zip(backbones, base_out_sizes)])

    def forward(self, batch):
        (l, r), m, qs = batch
        probs = []
        for i in range(len(self.models)):
            probs.append(self.models[i](((l[i], r[i]), m, qs)))
        probs = torch.cat(probs, 1)

        if self.training:
            temp = 0.1
            weights = torch.softmax(probs / temp, 1)
            return torch.sum(weights * probs, 1, keepdim=True)
        else:
            return torch.max(probs, dim=1, keepdim=True).values


def get_reti_pioneer(fast: bool):
    if fast:
        backbones = [nn.Identity(), nn.Identity(), nn.Identity()]
    else:
        backbones = [get_RETFound(), get_SwinB(), get_VimS()]
    base_out_sizes = [1024, 1024, 384]
    mid_size = 256
    fuse_dim = 256
    n_meta = 3 + 7
    num_classes = 1
    return ComplexModel(backbones, base_out_sizes, mid_size, fuse_dim, n_meta, num_classes)
