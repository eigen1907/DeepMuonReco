from logging import getLogger
from pathlib import Path
from functools import cached_property
from typing import Any, Callable
from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.states import RunningStage
from hydra.utils import get_class
from torch.utils.data import DataLoader
from ..utils.logging import elapsed_timer


_logger = getLogger(__name__)

__all__ = [
    'DataModule',
]


class DataModule(LightningDataModule):


    def __init__(
        self,
        dataset: str,
        root: str,
        #
        tracker_track_feature_list: list[str],
        dt_segment_feature_list: list[str],
        csc_segment_feature_list: list[str],
        rpc_hit_feature_list: list[str],
        gem_hit_feature_list: list[str],
        train_file: str | None = None,
        val_file: str | None = None,
        test_file: str | None = None,
        predict_file: str | None = None,
        #
        batch_size: int = 256,
        eval_batch_size: int = 256,
        train_sampler: Callable | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        #
        train_max_events: int | None = None,
        val_max_events: int | None = None,
        test_max_events: int | None = None,
        predict_max_events: int | None = None,
    ) -> None:
        """
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

    def get_dataset(self, stage: RunningStage):
        prefix = stage.dataloader_prefix
        if prefix is None:
            raise RuntimeError(f'got an unexpected {stage=}')

        dataset_cls = get_class(self.hparams['dataset'])
        _logger.info(f'Using dataset class {dataset_cls.__name__} for {prefix} set')
        file_name = self.hparams[f'{prefix}_file']
        if file_name is None:
            raise ValueError(f'File name for {prefix} set is not specified.')

        path = Path(self.hparams['root']) / file_name
        if not path.exists():
            raise FileNotFoundError(f'File {path} does not exist.')
        _logger.info(f'Loading {prefix} set from {path}')

        kwargs = {}
        key_list = [
            'tracker_track_feature_list',
            'dt_segment_feature_list',
            'csc_segment_feature_list',
            'rpc_hit_feature_list',
            'gem_hit_feature_list',
        ]
        for key in key_list:
            kwargs[key] = self.hparams[key]

        with elapsed_timer() as elapsed_time:
            dataset = dataset_cls(
                path=path,
                max_events=self.hparams[f'{prefix}_max_events'],
                **kwargs,
            )

        # logging the dataset, the number of examples  for debugging
        _logger.info(f'Loaded {prefix} set with {len(dataset)} examples in {elapsed_time():.2f} seconds')

        return dataset

    @cached_property
    def train_set(self):
        return self.get_dataset(RunningStage.TRAINING)

    @cached_property
    def val_set(self):
        return self.get_dataset(RunningStage.VALIDATING)

    @cached_property
    def test_set(self):
        return self.get_dataset(RunningStage.TESTING)

    @cached_property
    def predict_set(self):
        return self.get_dataset(RunningStage.PREDICTING)

    def train_dataloader(self):
        dataset = self.train_set

        kwargs: dict[str, Any] = {}
        if train_sampler := self.hparams.get('train_sampler', None):
            kwargs['sampler'] = train_sampler(dataset=dataset)
        else:
            kwargs['shuffle'] = True

        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams['batch_size'],
            drop_last=True,
            num_workers=self.hparams['num_workers'],
            pin_memory=self.hparams['pin_memory'],
            collate_fn=dataset.collate,
            **kwargs,
        )

    def _eval_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams['eval_batch_size'],
            num_workers=self.hparams['num_workers'],
            pin_memory=self.hparams['pin_memory'],
            collate_fn=dataset.collate,
        )

    def val_dataloader(self):
        _logger.info('Creating validation dataloader')
        return self._eval_dataloader(self.val_set)

    def test_dataloader(self):
        _logger.info('Creating test dataloader')
        return self._eval_dataloader(self.test_set)

    def predict_dataloader(self):
        _logger.info('Creating prediction dataloader')
        return self._eval_dataloader(self.predict_set)

    def setup(self, stage: str):
        _logger.info(f'Setting up {stage=}')
        if stage == 'fit':
            self.train_set
            self.val_set
        elif stage == 'validate':
            self.val_set
        elif stage == 'test':
            self.test_set
        elif stage == 'predict':
            self.predict_set
        else:
            raise RuntimeError(f'got an unexpected {stage=}')

    def teardown(self, stage: str):
        _logger.info(f'Tearing down {stage=}')
        if stage == 'fit':
            delattr(self, 'train_set')
            delattr(self, 'val_set')
        elif stage == 'validate':
            # delattr(self, 'val_set')
            ...
        elif stage == 'test':
            delattr(self, 'test_set')
        elif stage == 'predict':
            delattr(self, 'predict_set')
        else:
            raise RuntimeError(f'got an unexpected {stage=}')
