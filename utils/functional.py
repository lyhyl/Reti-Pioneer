from typing import Optional, Sequence, Union

import torch
from ignite.utils import convert_tensor


def ukb_pb(batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False, **kwargs):
    l, r, m, q, y = batch
    ret = []
    if kwargs.get("cat_lr", True):
        x = convert_tensor(torch.concat([l, r], 1), device=device, non_blocking=non_blocking)
    else:
        x = (convert_tensor(l, device=device, non_blocking=non_blocking),
             convert_tensor(r, device=device, non_blocking=non_blocking))
    ret.append(x)
    if kwargs.get("meta", True):
        m = convert_tensor(m.float(), device=device, non_blocking=non_blocking)
        ret.append(m)
    if kwargs.get("quality", True):
        q = (convert_tensor(q[0].float(), device=device, non_blocking=non_blocking),
             convert_tensor(q[1].float(), device=device, non_blocking=non_blocking))
        ret.append(q)
    if kwargs.get("ys", None) is not None:
        y = convert_tensor(y[kwargs.get("ys")].float(), device=device, non_blocking=non_blocking)
    else:
        y = convert_tensor(y.float(), device=device, non_blocking=non_blocking)
    return ret if len(ret) > 1 else ret[0], y


def pb_l_r_m_q_y(batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False):
    return ukb_pb(batch, device, non_blocking, cat_lr=False, meta=True, quality=True, ys=None)
