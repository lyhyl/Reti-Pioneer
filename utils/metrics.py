from typing import Callable, Tuple, Union

import ignite.metrics
import torch


def _TFPN(cm: torch.Tensor) -> Tuple[torch.Tensor]:
    whole = torch.sum(cm)
    col = torch.sum(cm, dim=0)
    row = torch.sum(cm, dim=1)
    diag = torch.diag(cm)

    TP = diag
    TN = whole - col - row + diag
    FP = col - diag
    FN = row - diag

    return TP, TN, FP, FN


def _GMean(cm: torch.Tensor, average: bool) -> Union[torch.Tensor, float]:
    TP, TN, FP, FN = _TFPN(cm)
    recall = TP / (TP + FN)
    spec = TN / (FP + TN)
    gmean = torch.sqrt(torch.mul(recall, spec))
    if average:
        return torch.mean(gmean).item()
    else:
        return gmean


def _SEN(cm: torch.Tensor, average: bool) -> Union[torch.Tensor, float]:
    TP, TN, FP, FN = _TFPN(cm)
    sen = TP / (TP + FN)
    if average:
        return torch.mean(sen).item()
    else:
        return sen


def _SPE(cm: torch.Tensor, average: bool) -> Union[torch.Tensor, float]:
    TP, TN, FP, FN = _TFPN(cm)
    spe = TN / (TN + FP)
    if average:
        return torch.mean(spe).item()
    else:
        return spe


def _PPV(cm: torch.Tensor, average: bool) -> Union[torch.Tensor, float]:
    TP, TN, FP, FN = _TFPN(cm)
    ppv = TP / (TP + FP)
    if average:
        return torch.mean(ppv).item()
    else:
        return ppv


def _NPV(cm: torch.Tensor, average: bool) -> Union[torch.Tensor, float]:
    TP, TN, FP, FN = _TFPN(cm)
    npv = TN / (TN + FN)
    if average:
        return torch.mean(npv).item()
    else:
        return npv


def GMean(num_class: int,
          average: bool = True,
          output_transform: Callable = lambda x: x,
          device: Union[str, torch.device] = torch.device("cpu")) -> ignite.metrics.MetricsLambda:
    cm = ignite.metrics.ConfusionMatrix(num_class, None, output_transform, device)
    return ignite.metrics.MetricsLambda(_GMean, cm, average)


def SEN(num_class: int,
        average: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu")) -> ignite.metrics.MetricsLambda:
    cm = ignite.metrics.ConfusionMatrix(num_class, None, output_transform, device)
    return ignite.metrics.MetricsLambda(_SEN, cm, average)


def SPE(num_class: int,
        average: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu")) -> ignite.metrics.MetricsLambda:
    cm = ignite.metrics.ConfusionMatrix(num_class, None, output_transform, device)
    return ignite.metrics.MetricsLambda(_SPE, cm, average)


def PPV(num_class: int,
        average: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu")) -> ignite.metrics.MetricsLambda:
    cm = ignite.metrics.ConfusionMatrix(num_class, None, output_transform, device)
    return ignite.metrics.MetricsLambda(_PPV, cm, average)


def NPV(num_class: int,
        average: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu")) -> ignite.metrics.MetricsLambda:
    cm = ignite.metrics.ConfusionMatrix(num_class, None, output_transform, device)
    return ignite.metrics.MetricsLambda(_NPV, cm, average)


def BGMean(
        average: bool = True,
        device: Union[str, torch.device] = torch.device("cpu")) -> ignite.metrics.MetricsLambda:
    pass


def ErrorRatio() -> ignite.metrics.MetricsLambda:
    return ignite.metrics.MeanAbsoluteError()


def sensitivity_score_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    from imblearn.metrics import sensitivity_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return sensitivity_score(y_true, y_pred)


class SensitivityScore(ignite.metrics.EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        try:
            from imblearn.metrics import sensitivity_score  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This contrib module requires imblearn to be installed.")

        super(SensitivityScore, self).__init__(
            sensitivity_score_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
        )


def specificity_score_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    from imblearn.metrics import specificity_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return specificity_score(y_true, y_pred)


class SpecificityScore(ignite.metrics.EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        try:
            from imblearn.metrics import specificity_score  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This contrib module requires imblearn to be installed.")

        super(SpecificityScore, self).__init__(
            specificity_score_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
        )


def ap_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    from sklearn.metrics import average_precision_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return average_precision_score(y_true, y_pred)


class AveragePrecision(ignite.metrics.EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        try:
            from sklearn.metrics import average_precision_score  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This contrib module requires scikit-learn to be installed.")

        super(AveragePrecision, self).__init__(
            ap_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device
        )
