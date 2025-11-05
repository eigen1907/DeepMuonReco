from logging import getLogger
from typing import cast
from functools import partial
import torch
from typing import Any
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tensordict import TensorDict
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from torchmetrics import MetricCollection
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchmetrics import MetricCollection
from omegaconf import DictConfig
from aim.pytorch_lightning import AimLogger
from aim.storage.object import CustomObject
from hydra.utils import instantiate

# FROG
from .utils.config import build_tensordictmodule
from .utils.config import build_tensordictsequential
from .utils.config import build_metric_collection
from .utils.optim import group_parameters


_logger = getLogger(__name__)



class Model(LightningModule):

    def __init__(
        self,
        augmentation: TensorDictSequential,
        preprocessing: TensorDictSequential,
        postprocessing: TensorDictSequential,
        model: TensorDictModule,
        criterion_function: TensorDictSequential,
        criterion_reduction: TensorDictSequential,
        val_pre_metric_postprocessing: TensorDictSequential,
        val_metrics: MetricCollection,
        test_pre_metric_postprocessing: TensorDictSequential,
        test_metrics: MetricCollection,
        optimizer_config: DictConfig,
        lr_scheduler_config: DictConfig,
    ) -> None:
        super().__init__()

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.val_pre_metric_postprocessing = val_pre_metric_postprocessing
        self.test_pre_metric_postprocessing = test_pre_metric_postprocessing
        self.model = model
        self.criterion_function = criterion_function
        self.criterion_reduction = criterion_reduction
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self._optimizer_config = optimizer_config
        self._lr_scheduler_config = lr_scheduler_config

        self.lr = optimizer_config.lr

    @classmethod
    def from_config(cls, config: DictConfig):
        augmentation = build_tensordictsequential(config.augmentation)
        preprocessing = build_tensordictsequential(config.preprocessing)
        postprocessing = build_tensordictsequential(config.postprocessing)

        model = build_tensordictmodule(config.model)

        criterion_function = build_tensordictsequential(config.criterion.function)
        criterion_reduction = build_tensordictsequential(config.criterion.reduction)

        val_pre_metric_postprocessing = build_tensordictsequential(config.metric.val.pre_metric_postprocessing)
        val_metrics = build_metric_collection(config=config.metric.val.metric, stage=RunningStage.VALIDATING)

        test_pre_metric_postprocessing = build_tensordictsequential(config.metric.test.pre_metric_postprocessing)
        test_metrics = build_metric_collection(config=config.metric.test.metric, stage=RunningStage.TESTING)

        optimizer_config = instantiate(config.optimizer)
        lr_scheduler_config = instantiate(config.lr_scheduler)

        return cls(
            # data processing pipelines
            augmentation=augmentation,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            # model
            model=model,
            # criterion
            criterion_function=criterion_function,
            criterion_reduction=criterion_reduction,
            # metrics and postprocessing
            val_pre_metric_postprocessing=val_pre_metric_postprocessing,
            val_metrics=val_metrics,
            test_pre_metric_postprocessing=test_pre_metric_postprocessing,
            test_metrics=test_metrics,
            # optimizer and lr_scheduler configs
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
        )


    def forward(
        self,
        batch: TensorDict,
    ) -> TensorDict:
        """
        """
        with torch.no_grad():
            batch = self.preprocessing(batch)
        batch = self.model(batch)
        return batch

    def training_step(self, batch: TensorDict) -> Tensor:
        """
        """
        with torch.no_grad():
            batch = self.augmentation(batch)
        batch = self(batch)
        batch = self.criterion_function(batch)
        loss: Tensor = self.criterion_reduction(batch)['loss']
        self.log(name=f'{RunningStage.TRAINING.value}_loss', value=loss) # FIXME:
        return loss

    def _eval_step(
        self,
        batch: TensorDict,
        pre_metric_postprocessing: TensorDictSequential,
        metric_collection: MetricCollection,
    ) -> TensorDict:
        """
        """
        batch = self(batch)
        batch = self.criterion_function(batch)
        batch = self.postprocessing(batch)
        batch = pre_metric_postprocessing(batch)
        metric_collection.update(batch)
        return batch

    def validation_step(self, batch: TensorDict) -> TensorDict:
        return self._eval_step(
            batch=batch,
            pre_metric_postprocessing=self.val_pre_metric_postprocessing,
            metric_collection=self.val_metrics,
        )

    def test_step(self, batch: TensorDict) -> TensorDict:
        return self._eval_step(
            batch=batch,
            pre_metric_postprocessing=self.test_pre_metric_postprocessing,
            metric_collection=self.test_metrics,
        )

    def predict_step(self, batch: TensorDict) -> TensorDict:
        batch = self(batch)
        batch = self.postprocessing(batch)
        return batch

    def track(self, name: str, value: Any) -> None:
        if not isinstance(self.logger, AimLogger):
            # TODO: raise warning
            return
        name, context = self.logger.parse_context(name)
        self.logger.experiment.track(
            value=value,
            name=name,
            step=self.global_step,
            epoch=self.current_epoch,
            context=context
        )

    def _log(self, key: str, value: Any, log_dict: dict) -> None:
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._log(key=f'{key}_{sub_key}', value=sub_value, log_dict=log_dict)
        elif isinstance(value, CustomObject) and isinstance(self.logger, AimLogger):
            self.logger.experiment.track(value=value, name=key)
        else:
            log_dict[key] = value

    def _on_epoch_end(self, metrics: MetricCollection) -> None:
        log_dict = {}
        for key, value in metrics.compute().items():
            self._log(key=key, value=value, log_dict=log_dict)
        if len(log_dict) > 0:
            self.log_dict(log_dict)
        metrics.reset()

    def on_train_epoch_start(self) -> None:
        if isinstance(self.logger, AimLogger):
            self.track(value=self.current_epoch, name='epoch')

    def on_validation_epoch_end(self) -> None:
        return self._on_epoch_end(self.val_metrics)

    def on_test_epoch_end(self) -> None:
        return self._on_epoch_end(self.test_metrics)

    def _configure_optimizer(self) -> Optimizer:
        config = self._optimizer_config
        params = group_parameters(
            model=self.model,
            weight_decay=config.weight_decay,
        )

        optimizer = config.optimizer(
            params=params,
            lr=self.lr,
        )
        return optimizer

    def _configure_lr_scheduler(self, optimizer: Optimizer):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        _logger.info(f'Configuring LR scheduler with {self._lr_scheduler_config=}')

        config = self._lr_scheduler_config
        if config is None:
            return None

        scheduler_partial = cast(partial, config.scheduler)
        scheduler_kwargs: dict[str, Any] = {}

        if scheduler_partial.func is CosineAnnealingWarmRestarts:
            num_batches = self.trainer.estimated_stepping_batches
            _logger.info(f'Using estimated stepping batches of {num_batches} for CosineAnnealingWarmRestarts')
            scheduler_kwargs['T_0'] = int(config.t0 * num_batches)

        lr_scheduler_config: dict[str, Any] = {
            'scheduler': scheduler_partial(optimizer=optimizer, **scheduler_kwargs),
        }

        for key in ['interval', 'frequency', 'monitor', 'strict', 'name']:
            if value := config.get(key, None):
                lr_scheduler_config[key] = value

        return lr_scheduler_config


    def configure_optimizers(self):
        optimizer = self._configure_optimizer()

        _logger.info(f'{optimizer=}')

        if lr_scheduler_config := self._configure_lr_scheduler(optimizer=optimizer):
            _logger.info(f'{lr_scheduler_config=}')
            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config,
            }
        else:
            return optimizer

    @property
    def num_params(self) -> int:
        return sum(each.numel() for each in self.model.parameters() if each.requires_grad)
