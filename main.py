from datetime import datetime

import torch
from ignite.utils import to_onehot

from dataset.UKBDataset import UKBDatasetFast
from model.RetiPioneer import get_reti_pioneer
from utils.functional import pb_l_r_m_q_y
from utils.run import single_fastds_run


def pb(batch, device, non_blocking):
    (l, r), m, (ql, qr), y = batch
    # eth to one hot
    eth = m[:, -1]
    e = to_onehot(eth.long(), 7)
    m = torch.cat([m[:, :-1], e], 1)
    batch = (l, r), m, (ql, qr), y
    return pb_l_r_m_q_y(batch, device, non_blocking)


def main():
    device = torch.device(f"cuda")
    ftime = datetime.now().strftime("%Y%m%d%H%M%S")
    diseases = ["t2dm", "thyroid", "osteoporosis", "gout", "hyperlipemia", "hypertension"]
    clinic_variables = ["baselineage", "gender", "weight", "ethnicity"]
    dataset = UKBDatasetFast("data/UKBCompressed", None, clinic_variables, disease=[0], use_pretrain=["RETF", "SwinB", "VimS"])

    for i in diseases:
        for y in [0, 5, 10]:
            print(f"Training {i} at {y} years")
            model = get_reti_pioneer(True)
            model = model.to(device)
            dataset.set_target(y, [i])
            single_fastds_run(
                model,
                lr=1e-4, epochs=20, epochs_factor=2, warmup_lr=1e-6, warmup_epochs=4,
                tds=dataset, vds=None, xdss=[], pb=pb, bs=512,
                device=device,
                tbdir=f"ckpt/{ftime}/{i}/y{y}",
                save=True,
                save_prob=True
            )


if __name__ == "__main__":
    main()
