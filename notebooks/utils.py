import sys, os
sys.path.append('../src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import Sigmoid

from tensordict.nn import TensorDictModule, TensorDictSequential

from deepmuonreco.data import InnerTrackSelectionDataset
from deepmuonreco.nn import InnerTrackSelectionTransformer, Normalize
from deepmuonreco.nn.utils import init_params
from deepmuonreco.nn import SelectedBCEWithLogitsLoss

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from hist import Hist
from hist.intervals import clopper_pearson_interval
import hist

import mplhep as hep
import matplotlib as mpl
import uproot
import awkward as ak


def load_model(
    checkpoint_path,
    device,
    dim_model,
    dim_feedforward,
    num_heads,
    num_layers,
    norm_stats,
):  
    preprocessor = TensorDictSequential([
        TensorDictModule(
            module=Normalize(mean=norm_stats['track'][0], std=norm_stats['track'][1]),
            in_keys=['track'],
            out_keys=['track'],
        ),
        TensorDictModule(
            module=Normalize(mean=norm_stats['seg'][0], std=norm_stats['seg'][1]),
            in_keys=['seg'],
            out_keys=['seg'],
        ),
        TensorDictModule(
            module=Normalize(mean=norm_stats['hit'][0], std=norm_stats['hit'][1]),
            in_keys=['hit'],
            out_keys=['hit'],
        ),
    ])

    # actual model
    raw_model = TensorDictModule(
        module=InnerTrackSelectionTransformer(
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0.1,
        ),
        in_keys=[
            'track',
            ('pad_masks', 'track'),
            'seg',
            ('pad_masks', 'seg'),
            'hit',
            ('pad_masks', 'hit'),
        ],
        out_keys=[
            'logits'
        ],
    )

    raw_model.apply(init_params)

    # postprocessing: apply Sigmoid activation for score
    postprocessor = TensorDictSequential([
        TensorDictModule(
            module=Sigmoid(),
            in_keys=['logits'],
            out_keys=['score'],
        ),
    ])

    model = TensorDictSequential([
        preprocessor,
        raw_model,
        postprocessor,
    ])
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model

def get_targets_logits_scores(
    model,
    test_loader,
    device
):
    all_targets = []
    all_logits = []
    all_scores = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            
            mask = batch["pad_masks"]["track"]
            
            target = batch["target"]
            logits = outputs["logits"]
            scores = outputs["score"]
            
            valid_targets = target.masked_select(mask.logical_not()).cpu().numpy()
            valid_logits = logits.masked_select(mask.logical_not()).cpu().numpy()
            valid_scores = scores.masked_select(mask.logical_not()).cpu().numpy()
            
            all_targets.append(valid_targets)
            all_logits.append(valid_logits)
            all_scores.append(valid_scores)

    all_targets = np.concatenate(all_targets)
    all_logits = np.concatenate(all_logits)
    all_scores = np.concatenate(all_scores)

    return all_targets, all_logits, all_scores

def plot_hist_1d(ax, data, bins, weight=None, linewidth=2.5, yerr=True,
                 xlabel=None, ylabel=None, label=None, histtype="step", 
                 title=None, mask=None, color=None, density=None):
    if mask is None:
        mask = np.ones_like(data, dtype=bool)
    if weight is None:
        weight = np.ones_like(data)

    hist_1d = Hist(
        hist.axis.Regular(*bins),
        storage=hist.storage.Weight()
    )

    hist_1d.fill(data[mask], weight=weight[mask])
    hist_1d.plot(ax=ax, label=label, color=color, yerr=yerr,
                density=density, histtype=histtype, linewidth=linewidth)

    ax.set_xlim(bins[1], bins[2])
    ax.set_ylim(0, )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax, hist_1d

def plot_efficiency_1d(ax, num, denom, bins, xlabel=None, ylabel=None, label=None,
                      title=None, color=None, density=None):
    hist_num = Hist(
        hist.axis.Regular(*bins),
        storage=hist.storage.Weight()
    )
    hist_denom = Hist(
        hist.axis.Regular(*bins),
        storage=hist.storage.Weight()
    )

    hist_num.fill(num)
    hist_denom.fill(denom)

    eff = hist_num.to_numpy()[0] / hist_denom.to_numpy()[0]
    bin_edges = hist_num.axes[0].edges
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    lower, upper = clopper_pearson_interval(hist_num.to_numpy()[0], hist_denom.to_numpy()[0], 0.68)

    ax.errorbar(bin_centers, eff, yerr=[eff - lower, upper - eff], 
                fmt='o', label=label, color=color, 
                capsize=5, capthick=3, markersize=8,
                alpha=0.8)
    ax.set_xlim(bins[1], bins[2])
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax, eff


def calc_pt(px, py):
    return np.sqrt(px**2 + py**2)

def calc_eta(px, py, pz, eps=1e-9):
    p = np.sqrt(px**2 + py**2 + pz**2)
    return 0.5 * np.log((p + pz + eps) / (p - pz + eps))

def calc_phi(px, py):
    return np.arctan2(py, px)

def calc_track_pt(track_qoverp, track_lambda):
    track_p = 1.0 / np.abs(track_qoverp)
    track_pt = track_p * np.cos(track_lambda)
    return track_pt

def calc_track_eta(track_lambda):
    track_eta = -np.log(np.tan(0.5 * (np.pi/2 - track_lambda)))
    return track_eta

