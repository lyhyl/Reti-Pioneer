import os
from typing import Callable, List, Literal

import numpy as np
import torch
from skimage import draw
from torch.utils.data import Subset, random_split
from torchvision.transforms import Compose


class UKBDatasetFast:
    def __init__(self,
                 path: str,
                 transform: Compose = None,
                 meta: List[str] = [],
                 y: Literal[0, 5, 10] = 0,
                 disease: List[int] | List[str] = list(range(6)),
                 incident_exclude_prior: bool = True,
                 use_pretrain: Literal["RETF", "VimS", "VimT", "SwinB"] | None = None,
                 prefilter: Callable[["UKBDatasetFast"], np.ndarray] = lambda ds: ~np.isnan(ds.center),
                 get_index: bool = False) -> None:
        self.use_pretrain = use_pretrain
        if use_pretrain is None:
            data = np.load(os.path.join(path, "UKB_lr.npz"))
            self.xl: np.ndarray = data["xl"]
            self.xr: np.ndarray = data["xr"]
            size = self.xl.shape[-1]
            radius = size // 2
            self.mask = np.ones((size, size))
            self.mask[draw.disk((radius - .5, radius - .5), radius, shape=(size, size))] = 0
            self.xl[:, :, self.mask == 1] = 0
            self.xr[:, :, self.mask == 1] = 0
            del data
        elif use_pretrain == "RETF":
            data = np.load(os.path.join(path, "UKB_RETF.npz"))
            self.xl: np.ndarray = data["left"]
            self.xr: np.ndarray = data["right"]
            del data
        elif use_pretrain == "VimS":
            data = np.load(os.path.join(path, "UKB_vim_s.npz"))
            self.xl: np.ndarray = data["left"]
            self.xr: np.ndarray = data["right"]
            del data
        elif use_pretrain == "VimT":
            data = np.load(os.path.join(path, "UKB_vim_t.npz"))
            self.xl: np.ndarray = data["left"]
            self.xr: np.ndarray = data["right"]
            del data
        elif use_pretrain == "SwinB":
            data = np.load(os.path.join(path, "UKB_swin_v2_b.npz"))
            self.xl: np.ndarray = data["left"]
            self.xr: np.ndarray = data["right"]
            del data
        else:
            raise ValueError(f"unknown use_pretrain={use_pretrain}")

        data = np.load(os.path.join(path, "UKB_mqd.npz"))
        self.m: np.ndarray = data["m"]
        mn: List[str] = data["mn"].tolist()
        mi = [mn.index(n) for n in meta]
        self.m = self.m[:, mi]
        self.ql: np.ndarray = data["ql"]
        self.qr: np.ndarray = data["qr"]
        self.center: np.ndarray = data["center"]
        del data

        for i in [0, 5, 10]:
            data = np.load(os.path.join(path, f"UKB_y{i}.npz"))
            self.__setattr__(f"y{i}", data["y"])
            del data
        self.ys: List[np.ndarray] = [self.__getattribute__(f"y{i}") for i in [0, 5, 10]]
        self.prior_health = self.ys[0] == 0

        self.transform = transform
        self.indices: List[int] = []
        self.prefilter = prefilter
        self.set_target(y, disease, incident_exclude_prior)

        self.get_index = get_index

        self.disease_names = ["t2dm", "thyroid", "osteoporosis", "gout", "hyperlipemia", "hypertension"]

    def set_target(self, y: Literal[0, 5, 10], disease: List[int] | List[str], incident_exclude_prior: bool = True):
        if isinstance(disease[0], str):
            disease = [self.disease_names.index(n) for n in disease]
        self.disease = disease
        self.yy = self.ys[[0, 5, 10].index(y)]
        filter_init = self.prefilter(self)
        if incident_exclude_prior and y > 0:
            idx = filter_init
            for d in disease:
                idx = idx & self.prior_health[:, d]
            self.indices = np.flatnonzero(idx).tolist()
        else:
            self.indices = np.flatnonzero(filter_init).tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        index = self.indices[index]

        xl: np.ndarray = self.xl[index]
        xr: np.ndarray = self.xr[index]
        if self.use_pretrain is not None:
            l, r = xl, xr
        else:
            l, r = xl.transpose([1, 2, 0]), xr.transpose([1, 2, 0])
            if self.transform is not None:
                l, r = self.transform(l), self.transform(r)
        m = self.m[index]
        ql = self.ql[index]
        qr = self.qr[index]
        yy = self.yy[index][self.disease]

        ret = (l, r, m, (ql, qr), yy)
        if self.get_index:
            ret += (index,)

        return ret


def ukb_split(dataset: UKBDatasetFast,
              test_centers: List[Literal[0, 1, 2, 3, 4]],
              validate_ratio: float,
              generator: torch.Generator):
    """
    test_centers: map 0-4 -> 1-5 domain 5, 6 are combined as one
    """
    # 1,     2,     3,     4,      5,      6
    # 15223, 12464, 15743, 16235,  4864,   191,  (18 nan)
    center = dataset.center[dataset.indices] - 1
    center[center == 5] = 4

    test_cond = center == -1  # all False
    xdss: List[Subset] = []
    for tc in test_centers:
        tcc = center == tc
        test_cond |= tcc
        xdss.append(Subset(dataset, np.flatnonzero(tcc)))

    tv_idx = np.flatnonzero(~test_cond)
    tds, vds = random_split(Subset(dataset, tv_idx), [1 - validate_ratio, validate_ratio], generator)

    assert (len(tds) + len(vds) + sum(len(xds) for xds in xdss)) == len(dataset)

    return tds, vds, xdss
