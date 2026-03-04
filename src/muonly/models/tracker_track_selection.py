from logging import getLogger
from typing import Any
import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchmetrics import MeanMetric, MetricCollection

from .model import Model

__all__ = [
    'TrackerTrackSelectionModel',
]


_logger = getLogger(__name__)

from omegaconf import OmegaConf, ListConfig, DictConfig
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

        _logger.debug(f'Initializing {self.__class__.__name__} with')
        _logger.debug(f'  - net ({type(net)}): {net}')
        _logger.debug(f'  - in_keys ({type(in_keys)}): {in_keys}')
        _logger.debug(f'  - optim_config ({type(optim_config)}): {optim_config}')
        _logger.debug(f'  - pos_weight ({type(pos_weight)}): {pos_weight}')

        _logger.debug('Converting configuration values to standard Python containers if necessary...')
        in_keys = _to_container(in_keys)
        _logger.debug(f'  - in_keys after conversion: {in_keys}')

        net = TensorDictModule(
            module=net,
            in_keys=in_keys,
            out_keys=['logits'],
        )

        val_metrics = MetricCollection(
            metrics={
                'loss': MeanMetric(),
            }
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

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, *args: Tensor | None, **kwargs: Tensor | None) -> Tensor:
        logits = self.net.module(*args, **kwargs)
        scores = torch.sigmoid(logits)
        return scores

    def compute_loss(self, batch: TensorDict) -> Tensor:
        batch = self.net(batch)
        loss = self.criterion(
            input=batch['logits'],
            target=batch['target'].float(),
        )
        loss = loss[batch['tracker_track_data_mask']] # FIXME:

        return loss # return the unreduced loss for metric computation

    def _update_metrics(self, batch: TensorDict, metrics: MetricCollection):
        loss = self.compute_loss(batch)

        metrics['loss'].update(loss)

    def _compute_metrics(self, metrics: MetricCollection, stage_prefix: str) -> dict[str, Any]:
        log_dict = {}
        log_dict['loss'] = metrics['loss'].compute()
        return log_dict
