from datetime import datetime
from itertools import combinations

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor

from dataset.UKBDataset import UKBDatasetFast, ukb_split
from model.RETFound import get_RETFound_model
from model.RetiPioneer import get_reti_pioneer
from utils.functional import pb_l_r_m_q_y
from utils.run import single_fastds_run


def main(fast: bool):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    ftime = datetime.now().strftime("%Y%m%d%H%M%S")
    diseases = ["t2dm", "thyroid", "osteoporosis", "gout", "hyperlipemia", "hypertension"]
    clinic_variables = ["baselineage", "gender", "weight"]
    n_meta = len(clinic_variables)
    if fast:
        dataset = UKBDatasetFast("data/UKBCompressed", None, clinic_variables, disease=[0], use_pretrain="RETF")
        get_backbone = nn.Identity
    else:
        trans = Compose([ToPILImage(), ToTensor(), Normalize(0.3, 0.3, 0.3)])
        dataset = UKBDatasetFast("data/UKBCompressed", trans, clinic_variables, disease=[0], use_pretrain=None)
        get_backbone = get_RETFound_model

    for i in diseases:
        for y in [0, 5, 10]:
            for test in combinations(list(range(5)), 2):
                backbone = get_backbone()
                model = get_reti_pioneer(backbone, 1024, n_meta)
                model = model.to(device)

                dataset.set_target(y, [i])

                tds, vds, xdss = ukb_split(dataset, test, 0.1, torch.Generator().manual_seed(430))
                has_pos = [max([y for l, r, m, q, y in xds]) == 1 for xds in xdss]
                xdss = [xds for xds, hp in zip(xdss, has_pos) if hp]
                test_names = '_'.join((str(i) if hp else f"~{i}") for i, hp in zip(test, has_pos))
                if len(xdss) == 0:
                    print(i, test, "has no positive value, skip")
                    continue

                single_fastds_run(
                    model,
                    lr=1e-4, epochs=20, epochs_factor=2, warmup_lr=1e-6, warmup_epochs=4,
                    tds=tds, vds=vds, xdss=xdss, pb=pb_l_r_m_q_y, bs=(512 if fast else 4),
                    device=device,
                    tbdir=f"ckpt/{ftime}/{i}/y{y}/test_{test_names}",
                    save=True,
                    save_prob=True
                )


if __name__ == "__main__":
    main(True)