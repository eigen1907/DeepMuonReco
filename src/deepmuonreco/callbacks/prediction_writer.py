from pathlib import Path
from logging import getLogger
from torch import Tensor
import h5py as h5
from h5py import Dataset
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from hydra.core.hydra_config import HydraConfig


_logger = getLogger(__name__)


__all__ = [
    'PredictionWriter',
]


class PredictionWriter(BasePredictionWriter):

    def __init__(
        self,
        output_file_path: Path | None = None,
    ) -> None:
        super().__init__(write_interval='batch')

        self.output_file_path = output_file_path
        self.file: h5.File | None = None
        self.score_dataset: Dataset | None = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage == 'predict':
            if self.output_file_path is None:
                _logger.info("output_file_path is not set. Using default path from Hydra runtime config.")

                log_dir_path = Path(HydraConfig.get().runtime.output_dir)
                if not log_dir_path.exists():
                    raise FileNotFoundError(f'hydra:runtime.output_dir does not exist: {log_dir_path}')

                output_dir_path = log_dir_path / 'predict'
                output_dir_path.mkdir(parents=True, exist_ok=True)

                self.output_file_path = output_dir_path / 'output.h5'

            _logger.info(f'Creating prediction file: {self.output_file_path}')

            self.file = h5.File(self.output_file_path, 'w')

            self.score_dataset = self.file.create_dataset(
                name='score',
                shape=(0, ),
                maxshape=(None, ),
                dtype=h5.vlen_dtype(np.float32),
            )

    def extend(self, dataset: Dataset | None, chunk: Tensor) -> None:
        """
        """
        if dataset is None:
            raise RuntimeError(
                'Dataset is not initialized. '
                'Make sure to call setup() before writing predictions.'
            )
        batch_size = chunk.shape[0]
        new_dataset_size = dataset.shape[0] + batch_size
        dataset.resize(size=new_dataset_size, axis=0)
        dataset[-batch_size:] = chunk.cpu().numpy()

    def extend_vlen(self, dataset: Dataset | None, chunk: Tensor, mask: Tensor) -> None:
        """
        """
        if dataset is None:
            raise RuntimeError(
                'Dataset is not initialized. '
                'Make sure to call setup() before writing predictions.'
            )
        chunk_np = chunk.cpu().numpy()
        mask_np = mask.cpu().bool().numpy()

        chunk_np = [c[m] for c, m in zip(chunk_np, mask_np)]
        chunk_np = np.array(chunk_np, dtype=object)  # dtype=object for variable-length arrays

        batch_size = chunk.shape[0]
        new_dataset_size = dataset.shape[0] + batch_size
        dataset.resize(size=new_dataset_size, axis=0)
        dataset[-batch_size:] = chunk_np

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx
    ) -> None:
        self.extend_vlen(
            dataset=self.score_dataset,
            chunk=prediction['score'],
            mask=prediction['masks']['tracker_track'],
        )

    def on_predict_end(self, trainer, pl_module) -> None:
        if self.file is None:
            raise RuntimeError(
                'Prediction file is not initialized. '
                'Make sure to call setup() before writing predictions.'
            )
        _logger.info('Closing prediction file')
        self.file.close()
