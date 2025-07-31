from typing import List

import torch
import torch.nn as nn

class QualityAware(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 enable_q: bool,
                 learnable_q: bool):
        super().__init__()

        self.enable_q = enable_q
        if self.enable_q:
            self.q_fc = nn.Linear(3, 1)
            if not learnable_q:
                self.q_fc.requires_grad_(False)
                self.q_fc.weight[0, 0] = 1
                self.q_fc.weight[0, 1] = .5
                self.q_fc.weight[0, 2] = 0
                self.q_fc.bias[0] = 0
            self.q_fuse = nn.Bilinear(in_dim + 1, 2, out_dim, False)
            self.post_q = nn.Sequential(nn.SELU(True), nn.Linear(out_dim, out_dim))
        else:
            self.pre_m = nn.Sequential(nn.SELU(True), nn.Linear(in_dim, out_dim))

    def forward(self, xf: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        dev = q.device
        bat = q.shape[0]
        if self.enable_q:
            xf = torch.cat([torch.ones(bat, 1, device=dev), xf], dim=1)
            q = torch.cat([torch.ones(bat, 1, device=dev), self.q_fc(q)], dim=1)
            xqf = self.q_fuse(xf, q)
            return self.post_q(xqf)
        else:
            return self.pre_m(xf)
