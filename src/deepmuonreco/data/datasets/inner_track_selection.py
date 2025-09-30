from pathlib import Path
from logging import getLogger
import uproot
import h5py as h5
import numpy as np
import awkward as ak
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
from tensordict import pad_sequence


_logger = getLogger(__name__)


class InnerTrackSelectionDataset(Dataset):

    TRACK_FEATURE_LIST = [
        'px', 'py', 'pz',
        'vx', 'vy', 'vz',
        'charge',
        'chi2', 'ndof',
        'dsz', 'dsz_error',
        'dxy', 'dxy_error',
        'lambda', 'lambda_error',
        'phi', 'phi_error',
        'qoverp', 'qoverp_error',
    ]

    SEGMENT_FEATURE_LIST = [
        'pos_x', 'pos_y', 'pos_z',
        'pos_err_x', 'pos_err_y',
        'dir_x', 'dir_y', 'dir_z',
        'dir_err_x', 'dir_err_y',
        'chi2', 'dof',
    ]

    RECHIT_FEATURE_LIST = [
        'pos_x', 'pos_y', 'pos_z',
        'pos_err_x', 'pos_err_y',
        'bx', 'cls_size',
    ]

    DIM_TRACK = len(TRACK_FEATURE_LIST)
    DIM_SEGMENT = len(SEGMENT_FEATURE_LIST)
    DIM_RECHIT = len(RECHIT_FEATURE_LIST)
    DIM_TARGET = 1

    def __init__(
        self,
        path: str | Path,
        max_events: int | float | None = None,
        track_feature_list: list[str] | None = None,
        segment_feature_list: list[str] | None = None,
        hit_feature_list: list[str] | None = None,
    ):
        super().__init__()

        track_feature_list = track_feature_list or self.TRACK_FEATURE_LIST
        segment_feature_list = segment_feature_list or self.SEGMENT_FEATURE_LIST
        hit_feature_list = hit_feature_list or self.RECHIT_FEATURE_LIST

        _logger.info(f"Loading data from {path} ...")
        _logger.info(f"  Track features: {track_feature_list}")
        _logger.info(f"  Segment features: {segment_feature_list}")
        _logger.info(f"  Hit features: {hit_feature_list}")


        path = Path(path)

        if path.suffix == '.root':
            loader = self.from_root
        elif path.suffix in ['.h5', '.hdf5']:
            loader = self.from_hdf5
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.example_list = loader(
            path=path,
            max_events=max_events,
            track_feature_list=track_feature_list,
            segment_feature_list=segment_feature_list,
            hit_feature_list=hit_feature_list,
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
        max_events: int | float | None,
        track_feature_list: list[str],
        segment_feature_list: list[str],
        hit_feature_list: list[str],
        treepath: str = 'muons1stStep/event',
    ) -> list[TensorDict]:
        with uproot.open(path) as file:
            tree = file[treepath]
            entry_stop = cls._get_stop(max_events=max_events, total=tree.num_entries)
            chunk = tree.arrays(library='ak', entry_stop=entry_stop)

        track_count = ak.count(chunk.track_pt, axis=1)
        dt_segment_count = ak.count(chunk.dt_segment_direction_x, axis=1)
        csc_segment_count = ak.count(chunk.csc_segment_direction_x, axis=1)
        segment_count = dt_segment_count + csc_segment_count

        rpc_rechit_count = ak.count(chunk.rpc_rechit_position_x, axis=1)
        gem_rechit_count = ak.count(chunk.gem_rechit_position_x, axis=1)
        rechit_count = rpc_rechit_count + gem_rechit_count

        muon_obj_count = segment_count + rechit_count
        chunk = chunk[(track_count > 0) & (muon_obj_count > 0)]

        nevents = len(chunk.track_pt)
        track_list, segment_list, rechit_list, target_list = [], [], [], []
        for i in range(nevents):
            # --- tracks: [pt, eta, phi] to [px, py, eta]
            pt  = chunk.track_pt[i]
            phi = chunk.track_phi[i]
            eta = chunk.track_eta[i]
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            track_arr = np.stack([px, py, eta], axis=1)  # (n_trk, 3)
            track_list.append(torch.tensor(track_arr, dtype=torch.float))

            # --- segment (DT + CSC): stack [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]
            def build_segment(prefix: str):
                return np.stack([
                    getattr(chunk, f'{prefix}_position_x')[i],
                    getattr(chunk, f'{prefix}_position_y')[i],
                    getattr(chunk, f'{prefix}_position_z')[i],
                    getattr(chunk, f'{prefix}_direction_x')[i],
                    getattr(chunk, f'{prefix}_direction_y')[i],
                    getattr(chunk, f'{prefix}_direction_z')[i],
                ], axis=1) if len(getattr(chunk, f'{prefix}_position_x')[i]) > 0 else np.zeros((0, 6))
            dt_arr  = build_segment('dt_segment')
            csc_arr = build_segment('csc_segment')
            segment_arr = np.concatenate([dt_arr, csc_arr], axis=0)  # (n_dt+n_csc,6)
            segment_list.append(torch.tensor(segment_arr, dtype=torch.float))

            # --- rechit (RPC + GEM): stack ([pos_x, pos_y, pos_z])
            def build_rec(prefix: str):
                return np.stack([
                    getattr(chunk, f'{prefix}_position_x')[i],
                    getattr(chunk, f'{prefix}_position_y')[i],
                    getattr(chunk, f'{prefix}_position_z')[i],
                ], axis=1) if len(getattr(chunk, f'{prefix}_position_x')[i]) > 0 else np.zeros((0, 3))
            rpc_arr = build_rec('rpc_rechit')
            gem_arr = build_rec('gem_rechit')
            rec_arr = np.concatenate([rpc_arr, gem_arr], axis=0)  # (n_rpc+n_gem,3)
            rechit_list.append(torch.tensor(rec_arr, dtype=torch.float))

            # --- target (is_trackermuon): [is_trackermuon] -> [is_trackermuon]
            tgt = ak.to_numpy(chunk.is_tracker_muon[i])
            target_list.append(torch.tensor(tgt, dtype=torch.float))

        example_chunk: list[TensorDict] = []
        for track, seg, rec, tgt in zip(track_list, segment_list, rechit_list, target_list):
            example_chunk.append(
                TensorDict(
                    source=dict(
                        track=track,
                        segment=seg,
                        rechit=rec,
                        target=tgt,
                    ),
                    batch_size=[],
                )
            )
        return example_chunk


    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        max_events: int | float | None,
        track_feature_list: list[str],
        segment_feature_list: list[str],
        hit_feature_list: list[str],
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
            total = len(file[next(iter(file.keys()))])
            stop = cls._get_stop(max_events=max_events, total=total)

            chunk = {}

            # NOTE: reconstructed tracks in the tracker
            chunk['track'] = [
                file[f'track_{each}'][:stop]
                for each in track_feature_list
            ]

            # NOTE: reconstructed segments in the muon system
            chunk['segment'] = []
            for feature in segment_feature_list:
                x = [
                    file[f'{prefix}_seg_{feature}'][:stop]
                    for prefix in ['dt', 'csc']
                ]
                x = [np.concat(each) for each in zip(*x)]
                chunk['segment'].append(x)

            # NOTE: reconstructed hits in the muon system
            chunk['rechit'] = []
            for feature in hit_feature_list:
                x = [
                    file[f'{prefix}_hit_{feature}'][:stop]
                    for prefix in ['rpc', 'gem']
                ]
                x = [np.concat(each) for each in zip(*x)]
                chunk['rechit'].append(x)

            chunk['target'] = [
                torch.from_numpy(each.astype(np.float32))
                for each in file['is_tracker_muon'][:stop]
            ]

        for key in ['track', 'segment', 'rechit']:
            chunk[key] = stack_features(chunk[key])

        return [
            TensorDict(dict(zip(chunk.keys(), each)))
            for each in zip(*chunk.values())
        ]



    @classmethod
    def collate(cls, example_list: list[TensorDict]) -> TensorDict:
        batch = pad_sequence(example_list, return_mask=True)
        batch['pad_masks'] = TensorDict(
            source=dict(
                track=batch['masks']['track'].logical_not(),
                segment=batch['masks']['segment'].logical_not(),
                rechit=batch['masks']['rechit'].logical_not(),
            ),
            batch_size=batch.batch_size,
        )
        return batch
