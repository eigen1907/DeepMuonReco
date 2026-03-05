# CLAUDE.md

## Project Overview

**muonly** — Muon detector tracker track selection using transformer models. Built with PyTorch Lightning + Hydra configuration.

Python >= 3.12, managed with `uv`.

## Environment Setup

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
  callbacks/           # Lightning callbacks
  data/                # Data configs (p, pq, pqv feature sets)
  datamodule/          # DataModule config
  debug/               # Debug presets: sanity-check, overfit, fdr, test
  hydra/               # Hydra runtime config
  logger/              # aim, csv loggers
  model/               # LightningModule config
  net/                 # Network architectures: latent_attention, vanilla_transformer
  optim/               # Optimizer + scheduler
  paths/               # Directory paths
  run/                 # Run behavior (fit, test, predict flags)
  torch/               # PyTorch settings (threads, matmul precision)
  trainer/             # Trainer configs: gpu, gpu_16-mixed, cpu, ddp, mps

src/muonly/
  models/              # LightningModule (model.py, tracker_track_selection.py)
  nn/
    models/            # Network architectures (latent_attention, vanilla_transformer)
    transformers/      # Transformer components (attention, mlp, perceiver)
    processings/       # Input preprocessing (normalize, index, log)
    utils.py           # Weight initialization
  data/
    datamodule.py      # Lightning DataModule
    datasets/          # Dataset implementations
  callbacks/           # Custom callbacks (PredictionWriter, etc.)
  metrics/             # Custom metrics (histogram)
  utils/               # Config, logging, optimizer utilities
```

## Configuration System

Hydra with config groups. Main config: `config/tts.yaml`. Override with CLI:

```bash
./train.py net=vanilla_transformer trainer=cpu debug=sanity-check
```

Config groups map to subdirectories under `config/`. The `debug` group is nullable (`debug: null` by default).

Custom OmegaConf resolvers registered in `train.py`:
- `${slug:2}` — random human-readable name (via coolname)
- `${len:expr}` — Python builtin `len()` function for calculating lengths
- `${randbits:32}` — random bits for seed generation

## Conventions

- Source code in `src/muonly/`, installed as editable package
- Type hints used throughout
- Models are LightningModules instantiated via `hydra.utils.instantiate`
- Network architectures live in `src/muonly/nn/models/`, separate from LightningModule wrappers in `src/muonly/models/`
- Logging via Python `logging` module
- Experiment tracking with Aim
- Data format: HDF5 (h5py) and ROOT (uproot)
