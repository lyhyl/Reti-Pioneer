import os
from typing import Callable, List, Literal

import numpy as np
from skimage import draw
from torchvision.transforms import Compose


type PretrainType = Literal["RETF", "VimS", "VimT", "SwinB"]


class UKBDatasetFast:
    def __init__(self,
                 path: str,
                 transform: Compose = None,
                 meta: List[str] = [],
                 y: Literal[0, 5, 10] = 0,
                 disease: List[int] = list(range(6)),
                 incident_exclude_prior: bool = True,
                 use_pretrain: PretrainType | List[PretrainType] | None = None,
                 prefilter: Callable[["UKBDatasetFast"], np.ndarray] = lambda ds: ~np.isnan(ds.center),
                 get_index: bool = False) -> None:
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
            self.use_pretrain = None
        else:
            if not isinstance(use_pretrain, list):
                use_pretrain = [use_pretrain]
            self.use_pretrain = use_pretrain
            self.xl: List[np.ndarray] = []
            self.xr: List[np.ndarray] = []
            ds_map = {
                "RETF": "UKB_RETF.npz",
                "VimS": "UKB_vim.npz",
                "SwinB": "UKB_swin.npz",
            }
            for pt in use_pretrain:
                data = np.load(os.path.join(path, ds_map[pt]))
                self.xl.append(data["left"].squeeze())
                self.xr.append(data["right"].squeeze())
                del data
                
        data = np.load(os.path.join(path, "UKB_mqd.npz"))
        self.m: np.ndarray = data["m"].astype(np.float32)
        mn: List[str] = data["mn"].tolist()
        self.ethnicity: np.ndarray = self.m[:, mn.index("ethnicity")].astype(np.int32)
        mi = [mn.index(n) for n in meta]
        self.m = self.m[:, mi]
        # mean = self.m.mean(0)
        # std = self.m.std(0)
        # if "gender" in meta:
        #     mean[meta.index("gender")] = 0.5
        #     std[meta.index("gender")] = 0.5
        # if "ethnicity" in meta:
        #     mean[meta.index("ethnicity")] = 0
        #     std[meta.index("ethnicity")] = 1
        # self.m = (self.m - mean) / std
        self.ql: np.ndarray = data["ql"].astype(np.float32)
        self.qr: np.ndarray = data["qr"].astype(np.float32)
        self.center: np.ndarray = data["center"].flatten()
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
        self.disease_names = ["t2dm", "thyroid", "osteoporosis", "gout", "hyperlipemia", "hypertension"]
        self.set_target(y, disease, incident_exclude_prior)

    def set_target(self, y: Literal[0, 5, 10], disease: List[int] | List[str], incident_exclude_prior: bool = True):
        if len(disease) > 0 and isinstance(disease[0], str):
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

        if self.use_pretrain is not None:
            if len(self.use_pretrain) == 1:
                xl: np.ndarray = self.xl[0][index]
                xr: np.ndarray = self.xr[0][index]
                l, r = xl, xr
            else:
                l = []
                r = []
                for i in range(len(self.use_pretrain)):
                    xl: np.ndarray = self.xl[i][index]
                    xr: np.ndarray = self.xr[i][index]
                    l.append(xl)
                    r.append(xr)
        else:
            xl: np.ndarray = self.xl[index]
            xr: np.ndarray = self.xr[index]
            l, r = xl.transpose([1, 2, 0]), xr.transpose([1, 2, 0])
            if self.transform is not None:
                l, r = self.transform(l), self.transform(r)
        m = self.m[index]
        ql = self.ql[index]
        qr = self.qr[index]
        yy = self.yy[index][self.disease]

        ret = ((l, r), m, (ql, qr), yy)

        return ret
