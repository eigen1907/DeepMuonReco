from typing import cast
import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
from tensordict import pad_sequence
import itertools


class InnerTrackSelectionDataset(Dataset):

    TRACK_FEATURE_LIST = [
        'qoverp', 'lambda', 'phi', 'dxy', 'dsz',
        'qoverp_error', 'lambda_error', 'phi_error', 'dxy_error', 'dsz_error',
        'chi2', 'ndof',
    ]

    SEGMENT_FEATURE_LIST = [
        'pos_x', 'pos_y', 'pos_z',
        'pos_err_x', 'pos_err_y',
        'dir_x', 'dir_y', 'dir_z',
        'dir_err_x', 'dir_err_y',
        'chi2', 'dof',
    ]

    HIT_FEATURE_LIST = [
        'pos_x', 'pos_y', 'pos_z',
        'pos_err_x', 'pos_err_y',
        'cls_size', 'bx',
    ]

    DIM_TRACK = len(TRACK_FEATURE_LIST)
    DIM_SEG = len(SEGMENT_FEATURE_LIST)
    DIM_HIT = len(HIT_FEATURE_LIST)
    DIM_TARGET = 1

    def __init__(self, path, treepath='muons1stStep/event', step=10000):
        """
        Memory-efficient dataset using uproot.iterate for large ROOT files.
        Keeps the same interface as the original version.
        """
        self.path = path
        self.treepath = treepath
        self.step = step
        self.example_list = self.process(path, treepath=treepath, step=step)

    def __getitem__(self, index: int) -> TensorDict:
        return self.example_list[index]

    def __len__(self) -> int:
        return len(self.example_list)

    @classmethod
    def process(cls, path: str, treepath: str = 'muons1stStep/event', step: int = 10000) -> list[TensorDict]:
        """
        Process ROOT file in chunks using uproot.iterate for better memory performance.
        """
        example_list = []

        for chunk in uproot.iterate(
            {path: treepath},
            step_size=step,
            library='ak',
        ):
            track_count = ak.count(chunk.track_px, axis=1)

            dt_seg_count = ak.count(chunk.dt_seg_pos_x, axis=1)
            csc_seg_count = ak.count(chunk.csc_seg_pos_x, axis=1)
            rpc_hit_count = ak.count(chunk.rpc_hit_pos_x, axis=1)
            gem_hit_count = ak.count(chunk.gem_hit_pos_x, axis=1)

            seg_count = dt_seg_count + csc_seg_count
            hit_count = rpc_hit_count + gem_hit_count

            valid = (track_count > 0) & ((seg_count > 0) | (hit_count > 0))
            chunk = chunk[valid]

            nevents = len(chunk.track_px)
            for i in range(nevents):
                n_seg = int(ak.num(chunk.dt_seg_pos_x[i], axis=0)) + int(ak.num(chunk.csc_seg_pos_x[i], axis=0))
                n_hit = int(ak.num(chunk.rpc_hit_pos_x[i], axis=0)) + int(ak.num(chunk.gem_hit_pos_x[i], axis=0))
                if (n_seg + n_hit) == 0:
                    continue
                if ak.num(chunk.track_qoverp[i], axis=0) == 0:
                    continue

                trk = ak.zip({
                    "qoverp": chunk.track_qoverp[i],
                    "lambda": chunk.track_lambda[i],
                    "phi": chunk.track_phi[i],
                    "dxy": chunk.track_dxy[i],
                    "dsz": chunk.track_dsz[i],
                    "qoverp_error": chunk.track_qoverp_error[i],
                    "lambda_error": chunk.track_lambda_error[i],
                    "phi_error": chunk.track_phi_error[i],
                    "dxy_error": chunk.track_dxy_error[i],
                    "dsz_error": chunk.track_dsz_error[i],
                    "chi2": chunk.track_chi2[i],
                    "ndof": chunk.track_ndof[i],
                })

                if ak.num(trk["qoverp"], axis=0) == 0:
                    continue
                trk_arr = np.stack([ak.to_numpy(trk[f]) for f in trk.fields], axis=1).astype(np.float32)
                trk_tensor = torch.from_numpy(trk_arr)

                # --- Segment (DT + CSC)
                def build_seg(prefix):
                    pos_x = getattr(chunk, f"{prefix}_pos_x")[i]
                    if ak.num(pos_x, axis=0) == 0:
                        return np.zeros((0, cls.DIM_SEG), np.float32)

                    arr = ak.zip({
                        "pos_x": pos_x,
                        "pos_y": getattr(chunk, f"{prefix}_pos_y")[i],
                        "pos_z": getattr(chunk, f"{prefix}_pos_z")[i],
                        "pos_err_x": getattr(chunk, f"{prefix}_pos_err_x")[i],
                        "pos_err_y": getattr(chunk, f"{prefix}_pos_err_y")[i],
                        "dir_x": getattr(chunk, f"{prefix}_dir_x")[i],
                        "dir_y": getattr(chunk, f"{prefix}_dir_y")[i],
                        "dir_z": getattr(chunk, f"{prefix}_dir_z")[i],
                        "dir_err_x": getattr(chunk, f"{prefix}_dir_err_x")[i],
                        "dir_err_y": getattr(chunk, f"{prefix}_dir_err_y")[i],
                        "chi2": getattr(chunk, f"{prefix}_chi2")[i],
                        "dof": getattr(chunk, f"{prefix}_dof")[i],
                    })
                    return np.stack([ak.to_numpy(arr[f]) for f in arr.fields], axis=1).astype(np.float32)

                seg_arr_dt = build_seg("dt_seg")
                seg_arr_csc = build_seg("csc_seg")
                seg_arr = np.concatenate([seg_arr_dt, seg_arr_csc], axis=0) if (seg_arr_dt.size + seg_arr_csc.size) > 0 else np.zeros((0, cls.DIM_SEG), np.float32)
                seg_tensor = torch.from_numpy(seg_arr)

                # --- Hit (RPC + GEM)
                def build_hit(prefix):
                    pos_x = getattr(chunk, f"{prefix}_pos_x")[i]
                    if ak.num(pos_x, axis=0) == 0:
                        return np.zeros((0, cls.DIM_HIT), np.float32)

                    arr = ak.zip({
                        "pos_x": pos_x,
                        "pos_y": getattr(chunk, f"{prefix}_pos_y")[i],
                        "pos_z": getattr(chunk, f"{prefix}_pos_z")[i],
                        "pos_err_x": getattr(chunk, f"{prefix}_pos_err_x")[i],
                        "pos_err_y": getattr(chunk, f"{prefix}_pos_err_y")[i],
                        "cls_size": getattr(chunk, f"{prefix}_cls_size")[i],
                        "bx": getattr(chunk, f"{prefix}_bx")[i],
                    })
                    return np.stack([ak.to_numpy(arr[f]) for f in arr.fields], axis=1).astype(np.float32)

                hit_arr_rpc = build_hit("rpc_hit")
                hit_arr_gem = build_hit("gem_hit")
                hit_arr = np.concatenate([hit_arr_rpc, hit_arr_gem], axis=0) if (hit_arr_rpc.size + hit_arr_gem.size) > 0 else np.zeros((0, cls.DIM_HIT), np.float32)
                hit_tensor = torch.from_numpy(hit_arr)

                # --- Target
                target = torch.tensor(ak.to_numpy(chunk.is_tracker_muon[i]), dtype=torch.float32)

                example_list.append(
                    TensorDict(
                        source=dict(
                            track=trk_tensor,
                            seg=seg_tensor,
                            hit=hit_tensor,
                            target=target,
                        ),
                        batch_size=[],
                    )
                )

        return example_list

    # -------------------------------
    # Collate
    # -------------------------------
    @classmethod
    def collate(cls, example_list: list[TensorDict]) -> TensorDict:
        """
        Merge a list of TensorDicts into a padded batch with masks.
        Compatible with the transformer model.
        """
        batch = pad_sequence(example_list, return_mask=True)
        batch['pad_masks'] = TensorDict(
            source=dict(
                track=batch['masks']['track'].logical_not(),
                seg=batch['masks']['seg'].logical_not(),
                hit=batch['masks']['hit'].logical_not(),
            ),
            batch_size=batch.batch_size,
        )
        return batch

    # -------------------------------
    # Normalization statistics
    # -------------------------------
    @classmethod
    def fit_normalization_stats(cls, dataset, max_events=None):
        means = {}
        M2 = {}
        counts = {}

        n_events = len(dataset) if max_events is None else min(len(dataset), max_events)

        for i, td in enumerate(dataset):
            if max_events is not None and i >= max_events:
                break

            for key in ["track", "seg", "hit"]:
                x = td[key]
                if x.numel() == 0:
                    continue
                x = x.float()
                batch_mean = x.mean(dim=0)
                batch_count = x.shape[0]

                if key not in means:
                    means[key] = batch_mean
                    M2[key] = torch.zeros_like(batch_mean)
                    counts[key] = batch_count
                else:
                    delta = batch_mean - means[key]
                    total = counts[key] + batch_count
                    means[key] += delta * batch_count / total
                    M2[key] += (x.var(dim=0, unbiased=False) * batch_count) + (delta**2) * counts[key] * batch_count / total
                    counts[key] = total

        stds = {k: torch.sqrt(M2[k] / counts[k]).clamp_min(1e-6) for k in means}
        return {k: (means[k].tolist(), stds[k].tolist()) for k in means}
