from pathlib import Path
from logging import getLogger
import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
from tensordict import pad_sequence


__all__ = [
    'TrackerTrackSelectionDataset',
]


_logger = getLogger(__name__)


class TrackerTrackSelectionDataset(Dataset):


    def __init__(
        self,
        path: str | Path,
        tracker_track_feature_list: list[str],
        dt_segment_feature_list: list[str],
        csc_segment_feature_list: list[str],
        rpc_hit_feature_list: list[str],
        gem_hit_feature_list: list[str],
        max_events: int | float | None = None,
    ) -> None:
        super().__init__()

        _logger.info(f"Loading data from {path} ...")
        _logger.info(f"  - Tracker track features: {tracker_track_feature_list}")
        _logger.info(f"  - DT segment features: {dt_segment_feature_list}")
        _logger.info(f"  - CSC segment features: {csc_segment_feature_list}")
        _logger.info(f"  - RPC hit features: {rpc_hit_feature_list}")
        _logger.info(f"  - GEM hit features: {gem_hit_feature_list}")

        path = Path(path)

        if path.suffix == '.root':
            loader = self.from_root
        elif path.suffix in ['.h5', '.hdf5']:
            loader = self.from_hdf5
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.example_list = loader(
            path=path,
            tracker_track_feature_list=tracker_track_feature_list,
            dt_segment_feature_list=dt_segment_feature_list,
            csc_segment_feature_list=csc_segment_feature_list,
            rpc_hit_feature_list=rpc_hit_feature_list,
            gem_hit_feature_list=gem_hit_feature_list,
            max_events=max_events,
        )

    def __getitem__(self, index: int) -> TensorDict:
        return self.example_list[index]

    def __len__(self) -> int:
        return len(self.example_list)


    @classmethod
    def _get_stop(cls, max_events: int | float | None, total: int) -> int | None:
        if max_events is None:
            stop = None
        elif isinstance(max_events, int):
            if max_events < 0:
                raise ValueError("max_events must be a non-negative integer or None.")
            stop = min(max_events, total)
        elif isinstance(max_events, float):
            if not (0.0 < max_events <= 1.0):
                raise ValueError("max_events must be a float in the range (0, 1].")
            stop = int(total * max_events)
            if stop < 1:
                raise ValueError("max_events results in an empty dataset.")
        else:
            raise TypeError("max_events must be an int, float, or None.")

        return stop


    @classmethod
    def from_root(
        cls,
        path: str | Path,
        tracker_track_feature_list: list[str],
        dt_segment_feature_list: list[str],
        csc_segment_feature_list: list[str],
        rpc_hit_feature_list: list[str],
        gem_hit_feature_list: list[str],
        max_events: int | float | None,
        treepath: str = 'muons1stStep/event',
    ) -> list[TensorDict]:
        raise NotImplementedError("Root file loading is not implemented yet.")

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        max_events: int | float | None,
        tracker_track_feature_list: list[str],
        dt_segment_feature_list: list[str],
        csc_segment_feature_list: list[str],
        rpc_hit_feature_list: list[str],
        gem_hit_feature_list: list[str],
    ) -> list[TensorDict]:
        """
        For the HDF5 files, we assume that event cleaning has already been
        performed: events without tracker tracks or without segments/hits in
        the muon detector have been removed. We also assume that px and py of
        each track are precomputed and stored.
        """
        def stack_features(feature_list: list[np.ndarray]) -> list[torch.Tensor]:
            return [
                torch.from_numpy(np.stack(each, axis=1, dtype=np.float32))
                for each in zip(*feature_list)
            ]

        with h5.File(path, 'r') as file:
            total = len(file[next(iter(file.keys()))]) # type: ignore
            stop = cls._get_stop(max_events=max_events, total=total)

            chunk = {}

            # NOTE: reconstructed tracker tracks
            chunk['tracker_track'] = [
                file[f'track_{each}'][:stop] # type: ignore
                for each in tracker_track_feature_list
            ]

            # NOTE: reconstructed
            chunk['dt_segment'] = [
                file[f'dt_seg_{each}'][:stop] # type: ignore
                for each in dt_segment_feature_list
            ]

            chunk['csc_segment'] = [
                file[f'csc_seg_{each}'][:stop] # type: ignore
                for each in csc_segment_feature_list
            ]

            # NOTE: reconstructed hits in the muon system
            chunk['rpc_hit'] = [
                file[f'rpc_hit_{each}'][:stop] # type: ignore
                for each in rpc_hit_feature_list
            ]

            chunk['gem_hit'] = [
                file[f'gem_hit_{each}'][:stop] # type: ignore
                for each in gem_hit_feature_list
            ]

            chunk['target'] = [
                torch.from_numpy(each.astype(np.float32))
                for each in file['is_tracker_muon'][:stop] # type: ignore
            ]

        for key in ['tracker_track', 'dt_segment', 'csc_segment', 'rpc_hit', 'gem_hit']:
            chunk[key] = stack_features(chunk[key])

        return [
            TensorDict(dict(zip(chunk.keys(), each)))
            for each in zip(*chunk.values())
        ]


    @classmethod
    def collate(cls, example_list: list[TensorDict]) -> TensorDict:
        # FIXME: tensordict.pad_sequence is probably slower than using torch.nn.utils.rnn.pad_sequence manually
        return pad_sequence(example_list, return_mask=True)
