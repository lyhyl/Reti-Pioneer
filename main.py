from datetime import datetime
from itertools import combinations

import numpy as np
import torch

from dataset.UKBDataset import UKBDatasetFast, ukb_split
from model.RETFound import get_RETFound_model
from model.RetiPioneer import get_reti_pioneer
from utils.run import single_fastds_run
from utils.functional import pb_l_r_m_q_y


def main():
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    ftime = datetime.now().strftime("%Y%m%d%H%M%S")
    diseases = ["t2dm", "thyroid", "osteoporosis", "gout", "hyperlipemia", "hypertension"]
    clinic_variables = ["baselineage", "gender", "weight"]
    n_meta = len(clinic_variables)
    dataset = UKBDatasetFast("data/UKBCompressed", None, clinic_variables,
                             disease=[0], use_pretrain="RETF")

    for i in diseases:
        for y in [0, 1, 3, 5, 10]:
            for test in combinations(list(range(5)), 2):
                backbone = get_RETFound_model()
                model = get_reti_pioneer(backbone, 1024, n_meta)
                model = model.to(device)

                dataset.set_target(y, [i])

                tds, vds, xdss = ukb_split(dataset, test, 0.1, torch.Generator().manual_seed(430))
                check = [np.vstack([y for l, r, m, q, y in xds]).max() == 1 for xds in xdss]
                xdss = [xds for xds, c in zip(xdss, check) if c]
                test2 = [center for center, c in zip(test, check) if c]
                if len(test2) == 0:
                    print(i, test, "has no positive value, skip")
                    continue

                test_names = '_'.join((str(i) if c else f"~{i}") for i, c in zip(test, check))
                single_fastds_run(
                    model,
                    lr=1e-4, epochs=20, epochs_factor=2, warmup_lr=1e-6, warmup_epochs=4,
                    tds=tds, vds=vds, xdss=xdss, pb=pb_l_r_m_q_y, bs=512,
                    device=device,
                    tbdir=f"ckpt/{ftime}/{diseases[i]}/y{y}/test_{test_names}",
                    save=True,
                    save_prob=True,
                    rocloss=True,
                )


if __name__ == "__main__":
    main()