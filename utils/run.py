import os
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.metrics.roc_auc import ROC_AUC, RocCurve
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import CosineAnnealingScheduler, StepStateScheduler, create_lr_scheduler_with_warmup
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics import Accuracy, Loss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from utils.metrics import AveragePrecision, SensitivityScore, SpecificityScore


def save_py(py: List[Tuple[np.ndarray, np.ndarray]] , path: str):
    def to_cpu_numpy(t: torch.Tensor): return t.cpu().numpy()
    p, y = zip(*py)
    p = np.concatenate(list(map(to_cpu_numpy, p)))
    y = np.concatenate(list(map(to_cpu_numpy, y)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, p=p, y=y)


def single_fastds_run(model: nn.Module,
                      lr: float, epochs: int, epochs_factor: int,
                      warmup_lr: float, warmup_epochs: int,
                      tds, vds, xdss: list,
                      pb: Callable, bs: int, device, tbdir: str,
                      save: bool = False,
                      save_prob: bool = False,
                      eval_train: bool = False,
                      **args):
    loss = nn.BCEWithLogitsLoss()
    
    optim = AdamW(model.parameters(), lr, betas=args.get("betas", (0.9, 0.999)), weight_decay=args.get("weight_decay", 0.01))
    scheduler = CosineAnnealingScheduler(optim, "lr", lr, 0.0, epochs // epochs_factor)
    scheduler = create_lr_scheduler_with_warmup(scheduler, warmup_lr, warmup_epochs)

    if not isinstance(xdss, list):
        xdss = [xdss]
    sampler = None
    balance_sampler = args.get("balance_sampler", False)
    if balance_sampler:
        ys = [x[-1] for x in tds]
        pos = float(sum(ys))
        neg = float(len(ys) - pos)
        ws = [((1 / pos) if y == 1 else (1 / neg)) for y in ys]
        sampler = WeightedRandomSampler(ws, len(ys), generator=torch.Generator().manual_seed(430))
    train_loader = DataLoader(tds, batch_size=bs, shuffle=not balance_sampler, sampler=sampler, pin_memory=False)
    valid_loader = DataLoader(vds, batch_size=bs, shuffle=False, pin_memory=False)
    test_loaders = [DataLoader(xds, batch_size=bs, shuffle=False, pin_memory=False) for xds in xdss]

    skip_vds = np.vstack([y for l, r, m, q, y in vds]).max() == 0

    trainer = create_supervised_trainer(
        model, optim, loss, device,
        prepare_batch=pb, amp_mode="amp"
    )

    def binize(py): return (torch.round(F.sigmoid(py[0])), py[1])
    def sig(py): return ((F.sigmoid(py[0]), py[1]))
    metrics = {
        "loss": Loss(loss),
        "acc": Accuracy(binize),
        "sen": SensitivityScore(binize),
        "spe": SpecificityScore(binize),
        "auc": ROC_AUC(sig),
        "ap": AveragePrecision(sig),
    }

    evaluator = create_supervised_evaluator(model, metrics, device, prepare_batch=pb)
    validator = create_supervised_evaluator(model, metrics, device, prepare_batch=pb)
    testers = [create_supervised_evaluator(model, metrics, device, prepare_batch=pb) for _ in range(len(xdss))]

    if save_prob:
        eos_eval = EpochOutputStore(sig)
        eos_eval.attach(evaluator, "py")
        eos_vald = EpochOutputStore(sig)
        eos_vald.attach(validator, "py")
        eos_tests = [EpochOutputStore(sig) for _ in range(len(xdss))]
        for i in range(len(xdss)):
            eos_tests[i].attach(testers[i], "py")

    pbar = ProgressBar()
    pbar.attach(trainer)
    pbar.attach(validator)
    for tester in testers:
        pbar.attach(tester)

    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def train_epoch_completed(engine: Engine):
        if save:
            torch.save(model.state_dict(), os.path.join(tbdir, "ckpt", f"{engine.state.epoch}.pt"))
        if eval_train:
            evaluator.run(train_loader) # too slow
        if not skip_vds:
            validator.run(valid_loader)
            if save_prob:
                save_py(validator.state.py, os.path.join(tbdir, "ckpt", f"py_{engine.state.epoch}_val.npz"))
        for i, (tester, test_loader) in enumerate(zip(testers, test_loaders)):
            tester.run(test_loader)
            if save_prob:
                save_py(tester.state.py, os.path.join(tbdir, "ckpt", f"py_{engine.state.epoch}_test{i}.npz"))

    os.makedirs(tbdir, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=tbdir)
    tb_logger.attach(trainer, OptimizerParamsHandler(optim), Events.EPOCH_STARTED)
    for tag, engine in [("train", evaluator), ("valid", validator)] + [(f"test_{i}", tester) for i, tester in enumerate(testers)]:
        tb_logger.attach_output_handler(
            engine,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    if save:
        os.makedirs(os.path.join(tbdir, "ckpt"), exist_ok=True)

    trainer.run(train_loader, warmup_epochs + epochs - 1)
