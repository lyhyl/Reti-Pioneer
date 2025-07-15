from typing import List
import torch
import torch.nn as nn


def get_reti_pioneer(backbone: nn.Module, backbone_output_size: int, n_meta: int):
    mid_size = 256
    fuse_dim = 256
    model = nn.Sequential(
        FuseBase(backbone, backbone_output_size, n_meta, mid_size, fuse_dim, flip_r=False),
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

        self.enable_q = enable_q
        if self.enable_q:
            self.q_fc = nn.Linear(3, 1)
            if not learnable_q:
                self.q_fc.requires_grad_(False)
                self.q_fc.weight[0, 0] = 1
                self.q_fc.weight[0, 1] = .5
                self.q_fc.weight[0, 2] = 0
                self.q_fc.bias[0] = 0
            self.q_fuse = nn.Bilinear(base_out_size + 1, 2, self.fuse_dim, False)
            self.post_q = nn.Sequential(nn.SELU(True), nn.Linear(self.fuse_dim, self.fuse_dim))
        else:
            self.pre_m = nn.Sequential(nn.SELU(True), nn.Linear(base_out_size, self.fuse_dim))
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
            if self.enable_q:
                xf = torch.cat([torch.ones(bat, 1, device=dev), xf], dim=1)
                q = torch.cat([torch.ones(bat, 1, device=dev), self.q_fc(q)], dim=1)
                xqf = self.q_fuse(xf, q)
                xqf2.append(self.post_q(xqf))
            else:
                xqf2.append(self.pre_m(xf))
        xqf2 = torch.cat(xqf2, dim=1)
        m = torch.cat([torch.ones(bat, 1, device=dev), m], dim=1)
        xqmf = self.m_fuse(xqf2, m)

        return xqmf
