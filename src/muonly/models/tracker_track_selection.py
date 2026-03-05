from logging import getLogger
from typing import Any
import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchmetrics import MetricCollection
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryROC
from torchmetrics.classification import BinaryAUROC
import matplotlib.pyplot as plt
from aim.sdk.objects.image import Image
from omegaconf import OmegaConf, ListConfig, DictConfig

from .model import Model
from ..metrics import Histogram

__all__ = [
    "TrackerTrackSelectionModel",
]


_logger = getLogger(__name__)


def _to_container(x: Any):
    if isinstance(x, (ListConfig, DictConfig)):
        x = OmegaConf.to_container(x)
    return x


class TrackerTrackSelectionModel(Model):
    def __init__(
        self,
        net,
        in_keys,
        optim_config,
        pos_weight,
    ) -> None:

        _logger.debug(f"Initializing {self.__class__.__name__} with")
        _logger.debug(
            f"  - net ({type(net)})"
        )  # do not print the net itself to avoid cluttering the logs
        _logger.debug(f"  - in_keys ({type(in_keys)}): {in_keys}")
        _logger.debug(f"  - optim_config ({type(optim_config)}): {optim_config}")
        _logger.debug(f"  - pos_weight ({type(pos_weight)}): {pos_weight}")

        _logger.debug(
            "Converting configuration values to standard Python containers if necessary..."
        )
        in_keys = _to_container(in_keys)
        _logger.debug(f"  - in_keys after conversion: {in_keys}")

        net = TensorDictModule(
            module=net,
            in_keys=in_keys,
            out_keys=["logits"],
        )

        val_metrics = MetricCollection(
            metrics={
                "loss": MeanMetric(),
                "roc": BinaryROC(compute_on_cpu=True),
                "auroc": BinaryAUROC(compute_on_cpu=True),
                "score_muon": Histogram(bins=40, range=(0, 1)),
                "score_bkg": Histogram(bins=40, range=(0, 1)),
            },
        )

        test_metrics = val_metrics.clone()

        super().__init__(
            net=net,
            optim_config=optim_config,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )

        if (pos_weight is not None) and (not torch.is_tensor(pos_weight)):
            pos_weight = torch.tensor(pos_weight)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, *args: Tensor | None, **kwargs: Tensor | None) -> Tensor:
        logits = self.net.module(*args, **kwargs)
        scores = torch.sigmoid(logits)
        return scores

    def compute_loss(self, batch: TensorDict) -> Tensor:
        batch = self.net(batch)
        loss = self.criterion(
            input=batch["logits"],
            target=batch["target"].float(),
        )
        mask = batch["tracker_track_data_mask"]
        loss = loss[mask]
        return loss  # return the unreduced loss for metric computation

    def _update_metrics(self, batch: TensorDict, metrics: MetricCollection):
        loss = self.compute_loss(batch)
        mask = batch["tracker_track_data_mask"]
        scores = torch.sigmoid(batch["logits"][mask])
        target = batch["target"][mask].long()

        metrics["loss"].update(loss)
        metrics["roc"].update(preds=scores, target=target)
        metrics["auroc"].update(preds=scores, target=target)
        metrics["score_muon"].update(scores[target == 1])
        metrics["score_bkg"].update(scores[target == 0])

    def _compute_metrics(
        self, metrics: MetricCollection, stage_prefix: str
    ) -> dict[str, Any]:
        output = {}
        output["loss"] = metrics["loss"].compute()
        output["auroc"] = metrics["auroc"].compute()

        # NOTE:
        fpr, tpr, thresholds = metrics["roc"].compute()
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        tnr = 1 - fpr
        auc = metrics["auroc"].compute().item()

        fig, ax = plt.subplots()
        ax.plot(tpr, tnr, label=f"AUC={auc:.4f}")
        ax.plot([0, 1], [1, 0], label="Random", color="gray", linestyle="--")
        ax.set_xlabel("True Positive Rate")
        ax.set_ylabel("True Negative Rate")
        ax.legend()
        fig.tight_layout()
        output["roc_curve"] = Image(fig)

        # NOTE: model response
        fig, ax = plt.subplots()
        metrics["score_bkg"].plot(
            ax=ax, label="Background", density=True, color="tab:blue", lw=2
        )
        metrics["score_muon"].plot(
            ax=ax, label="Muon", density=True, color="tab:orange", lw=2
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        output["score"] = Image(fig)

        return output
