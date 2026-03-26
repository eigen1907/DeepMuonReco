import abc
from logging import getLogger
from typing import Any
from tensordict.nn import TensorDictModule
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from tensordict import TensorDict
from torchmetrics import MetricCollection
from aim.pytorch_lightning import AimLogger
from aim.storage.object import CustomObject
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# monster
from ..utils.optim import get_parameter_groups


_logger = getLogger(__name__)


class Model(LightningModule, abc.ABC):
    """ """

    def __init__(
        self,
        net: TensorDictModule,
        optim_config,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
    ) -> None:
        super().__init__()

        self.net = net

        optim_config = OmegaConf.to_container(optim_config, resolve=True)
        if not isinstance(optim_config, dict):
            raise ValueError(
                f"Expected optim_config to be a dict, got {type(optim_config)}"
            )
        self.optim_config = optim_config

        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    ############################################################################
    # Forward and evaluation steps
    ############################################################################
    def forward(self, *args: Tensor | None, **kwargs: Tensor | None) -> Tensor:
        """
        forward method simply delegates to the underlying net, which is expected to be a TensorDictModule,
        so that it can be easily exported to ONNX and TorchScript
        """
        return self.net.module(*args, **kwargs)

    @abc.abstractmethod
    def compute_loss(self, batch: TensorDict) -> Tensor: ...

    def training_step(self, batch: TensorDict) -> Tensor:
        """ """
        loss = self.compute_loss(batch)
        loss = loss.mean()
        self.log(name="train_loss", value=loss)
        return loss

    def _eval_step(
        self,
        batch: TensorDict,
        metrics: MetricCollection,
    ) -> TensorDict:
        """
        implementation of shared evaluation step for validation and testing
        """
        self._update_metrics(batch, metrics)
        return batch

    @abc.abstractmethod
    def _update_metrics(self, batch: TensorDict, metrics: MetricCollection): ...

    def _on_eval_epoch_end(
        self, metrics: MetricCollection, stage: RunningStage
    ) -> None:
        prefix = stage.dataloader_prefix
        if prefix is None:
            raise ValueError(f"dataloader_prefix is None for {stage=}")

        if self.trainer.sanity_checking:
            metrics.reset()
            return

        output = self._compute_metrics(metrics, prefix)

        log_dict = {}
        self._log(key=prefix, value=output, log_dict=log_dict)

        plt.close("all")  # close all figures to prevent memory leak in AimLogger

        self.log_dict(log_dict)
        metrics.reset()

    @abc.abstractmethod
    def _compute_metrics(
        self,
        metrics: MetricCollection,
        stage_prefix: str,
    ) -> dict[str, Any]: ...

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, self.val_metrics)

    def on_validation_epoch_end(self) -> None:
        return self._on_eval_epoch_end(
            metrics=self.val_metrics,
            stage=RunningStage.VALIDATING,
        )

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, self.test_metrics)

    def on_test_epoch_end(self) -> None:
        return self._on_eval_epoch_end(
            metrics=self.test_metrics,
            stage=RunningStage.TESTING,
        )

    def predict_step(self, batch: TensorDict) -> TensorDict:
        return self.net(batch)

    ############################################################################
    # Aim tracking
    ############################################################################

    def track(self, name: str, value: Any) -> None:
        if not isinstance(self.logger, AimLogger):
            raise TypeError(f"Expected logger to be AimLogger, got {type(self.logger)}")
        name, context = self.logger.parse_context(name)
        self.logger.experiment.track(
            value=value,
            name=name,
            step=self.global_step,
            epoch=self.current_epoch,
            context=context,
        )

    # FIXME: rename
    def _log(self, key: str, value: Any, log_dict: dict) -> None:
        if isinstance(value, CustomObject):
            if isinstance(self.logger, AimLogger):
                self.track(name=key, value=value)
            else:
                _logger.warning(
                    f"CustomObject {value} cannot be logged without AimLogger."
                )
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._log(key=f"{key}_{sub_key}", value=sub_value, log_dict=log_dict)
        else:
            log_dict[key] = value

    ############################################################################
    # Optimizer and LR scheduler configuration
    ############################################################################

    def _configure_optimizer(self):
        param_groups = get_parameter_groups(
            self,
            weight_decay=self.optim_config["weight_decay"],
        )
        optimizer = AdamW(
            params=param_groups,
            lr=self.optim_config["lr"],
            betas=(
                self.optim_config["beta1"],
                self.optim_config["beta2"],
            ),
        )
        return optimizer

    def _configure_lr_scheduler(self, optimizer: AdamW):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        total_steps = self.trainer.estimated_stepping_batches
        if not isinstance(total_steps, int):
            raise ValueError(
                f"Expected estimated_stepping_batches to be int, got {type(total_steps)}"
            )

        warmup_steps = self.optim_config["warmup"]["frac_steps"]
        if isinstance(warmup_steps, int):
            if not (0 <= warmup_steps < total_steps):
                raise ValueError(
                    f"Expected warmup_steps to be in [0, total_steps), got {warmup_steps}"
                )
        elif isinstance(warmup_steps, float):
            if not (0 <= warmup_steps < 1):
                raise ValueError(
                    f"Expected warmup_steps to be in [0, 1), got {warmup_steps}"
                )

            warmup_steps = int(warmup_steps * total_steps)
        else:
            raise TypeError(
                f"Expected warmup_steps to be int or float, got {type(warmup_steps)}"
            )

        cosine_steps = total_steps - warmup_steps

        _logger.info(
            f"Configuring LR scheduler with total_steps={total_steps}, warmup_steps={warmup_steps}, cosine_steps={cosine_steps}"
        )

        warmup = LinearLR(
            optimizer=optimizer,
            start_factor=self.optim_config["warmup"]["start_factor"],
            total_iters=warmup_steps,
        )

        eta_min = (
            self.optim_config["lr"] * self.optim_config["cosine"]["eta_min_factor"]
        )
        cosine = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cosine_steps,
            eta_min=eta_min,
        )

        lr_scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[
                warmup,
                cosine,
            ],
            milestones=[
                warmup_steps,
            ],
        )

        return dict(
            scheduler=lr_scheduler,
            interval="step",
            frequency=1,
        )

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()

        lr_scheduler_config = self._configure_lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    ############################################################################
    # Other utilities
    ############################################################################

    @property
    def num_parameters(self) -> int:
        return sum(each.numel() for each in self.parameters() if each.requires_grad)
