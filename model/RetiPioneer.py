from typing import List

import torch
import torch.nn as nn

from model.QualityAware import QualityAware


def get_reti_pioneer(backbone: nn.Module, backbone_output_size: int, n_meta: int, flip_r: bool=False):
    mid_size = 256
    fuse_dim = 256
    model = nn.Sequential(
        FuseBase(backbone, backbone_output_size, n_meta, mid_size, fuse_dim, flip_r=flip_r),
        nn.SiLU(True),
        nn.Linear(mid_size, 1)
    )
    return model


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
