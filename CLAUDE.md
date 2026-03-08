# CLAUDE.md

## Project Overview

**muonly** — Muon detector tracker track selection (CMS experiment) using transformer models for binary classification of tracker tracks (muon vs background). Built with PyTorch Lightning + Hydra configuration.

Python >= 3.12, managed with `uv`. Build backend: `uv_build`.

## Key Commands

```bash
# Sanity check (quick validation that everything works)
uv run python ./train.py debug=sanity-check

# Training with default config (DO NOT RUN)
# do not run this command because it takes a long time to run, use the sanity check command above instead
uv run python ./train.py

# Training with overrides (DO NOT RUN)
uv run python ./train.py net=vanilla_transformer model.pos_weight=100

# Submit to HTCondor cluster (DO NOT RUN)
uv run python ./submit.py -cn tts --gpus 1 --cpus 3

# Prediction from checkpoint (DO NOT RUN)
uv run python ./predict.py --ckpt path/to/checkpoint.ckpt --gpu 0

# Linting
uv run ruff check src/
uv run ruff format src/

# Aim experiment tracker UI
aim up --port <PORT>
```

## Project Structure

```
train.py              # Main training entry point (Hydra)
predict.py            # Inference from checkpoint
submit.py             # HTCondor cluster job submission

config/
  tts.yaml            # Main config (tracker track selection)
  tts_run3.yaml       # Alternative config
  callbacks/           # Lightning callbacks (checkpoint, early stopping, etc.)
  data/                # Feature definitions and preprocessing pipelines
  datamodule/          # DataModule config (paths, batch sizes)
  debug/               # Debug presets: sanity-check, overfit, fdr, test
  hydra/               # Hydra runtime config
  logger/              # aim, csv loggers
  model/               # LightningModule config
  net/                 # Network architectures: latent_attention, vanilla_transformer
  optim/               # Optimizer (AdamW) + scheduler (linear warmup + cosine)
  paths/               # Directory paths
  run/                 # Run behavior (fit, test, predict flags)
  torch/               # PyTorch settings (threads, matmul precision)
  trainer/             # Trainer configs: gpu, gpu_16-mixed, cpu, ddp, mps

src/muonly/
  models/              # LightningModule wrappers
    model.py           # Abstract Model base class
    tracker_track_selection.py  # TrackerTrackSelectionModel (loss, metrics, logging)
  nn/
    models/            # Network architectures
      vanilla_transformer.py  # Transformer decoder-based model
      latent_attention.py     # Perceiver-based model with learned latent codes
    transformers/      # Transformer building blocks
      attention.py     # CrossAttention, SelfAttention (scaled dot-product)
      mlp.py           # Feedforward block (Linear-GELU-Linear-Dropout)
      transformer.py   # Encoder/Decoder layers and stacks
      perceiver.py     # Perceiver encoder/processor/decoder
    processings/       # Input preprocessing transforms
      normalize.py     # MinMaxScaling, Normalize
      log.py           # SignedLog1p (sign-preserving log)
      index.py         # Indexable base class
    utils.py           # Weight initialization (Normal(0, 0.02))
  data/
    datamodule.py      # Lightning DataModule (lazy loading, train/val/test splits)
    datasets/
      tracker_track_selection.py  # HDF5 dataset with variable-length sequences
  callbacks/
    prediction_writer.py    # Writes predictions to HDF5
    cuda_memory_summary.py  # Logs GPU memory usage
    time_printer.py         # Logs timing events
  metrics/
    histogram.py       # Histogram metric with matplotlib plotting
  utils/
    logging.py         # Aim experiment tracking setup
    optim.py           # Optimizer parameter grouping (decay vs no-decay)
```

## Configuration System

Hydra with config groups. Main config: `config/tts.yaml`. Override with CLI:

```bash
uv run python ./train.py net=vanilla_transformer trainer=cpu debug=sanity-check
```

Config groups map to subdirectories under `config/`. The `debug` group is nullable (`debug: null` by default).

Custom OmegaConf resolvers registered in `train.py`:
- `${slug:2}` — random human-readable name (via coolname)
- `${len:expr}` — Python builtin `len()` function for calculating lengths
- `${randbits:32}` — random bits for seed generation

## Architecture

### Data Flow
```
HDF5 files → TrackerTrackSelectionDataset → collate (pad + mask) → DataModule
  ↓
tracker_track features → SignedLog1p → MinMaxScaling → embed
muon detector features → MinMaxScaling → embed
  ↓
Network (VanillaTransformer or LatentAttention) → per-track logits
  ↓
BCEWithLogitsLoss (pos_weight=150, masked to valid tracks)
```

### Networks
- **VanillaTransformer**: Tracker tracks attend to concatenated muon detector embeddings via transformer decoder (cross-attention + self-attention). Default: dim=64, heads=8, layers=4.
- **LatentAttention**: Perceiver-based. Learned latent codes compress inputs, cross-attend between track and muon detector latents, then decode back to per-track predictions. Default: dim=64, heads=4, track_latents=64, muon_det_latents=16.

### Key Design Decisions
- Pre-norm (LayerNorm before sublayers) for training stability
- TensorDict for structured batch data
- Einops for readable tensor reshaping
- Separate param groups: weight decay on Linear/Conv weights only

## Conventions

- Source code in `src/muonly/`, installed as editable package
- Type hints used throughout
- Models are LightningModules instantiated via `hydra.utils.instantiate`
- Network architectures live in `src/muonly/nn/models/`, separate from LightningModule wrappers in `src/muonly/models/`
- Logging via Python `logging` module
- Experiment tracking with Aim
- Data format: HDF5 (h5py)
- No test suite — use `debug=sanity-check` for validation
- Linting: ruff (configured in pyproject.toml, excludes notebooks, allows F403 in `__init__.py`)

## Known Issues

- `pyproject.toml` description is placeholder text
- ROOT file loading path exists in dataset code but is not implemented
- Open TODOs/FIXMEs in: `model.py:147`, `optim.py:79`, `perceiver.py:206`, `latent_attention.py:14`, `submit.py:103`
