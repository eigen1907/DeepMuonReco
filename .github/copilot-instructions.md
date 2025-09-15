# DeepMuonReco - Deep Learning for Muon Reconstruction

DeepMuonReco is a PyTorch-based deep learning project for muon track selection and reconstruction using transformer models. It uses Hydra for configuration management and Aim for experiment tracking.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- **NEVER CANCEL** environment creation - takes 15-20 minutes. Set timeout to 30+ minutes.
- Create conda environment: `conda env create -f ./environment.yaml`
  - Note: Original environment.yaml uses `pytorch-gpu` which requires CUDA. For CPU testing, use `pytorch` instead.
  - Environment creation takes 15-20 minutes. NEVER CANCEL. Wait for completion.
- Alternative with micromamba: `micromamba create -y -f ./environment.yaml`
  - micromamba may fail due to SSL certificate issues in some environments.
  - If micromamba fails, use conda instead.

### Environment Activation  
- **Bash**: `source setup.sh` (requires micromamba) OR manually:
  ```bash
  source /usr/share/miniconda/etc/profile.d/conda.sh
  conda activate deepmuonreco-py312
  export PROJECT_ROOT=$(pwd)  # CRITICAL: Required for Hydra config resolution
  export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH}
  ```
- **Fish**: `source setup.fish`

### Build and Test
- No traditional build step required - this is a Python package.
- **Sanity check**: `./train.py --help` - should show Hydra configuration options.
- **Quick test run** (takes ~15 seconds): 
  ```bash
  ./train.py data.train_set.path=data/sanity-check.root data.val_set.path=data/sanity-check.root data.test_set.path=data/sanity-check.root data.train_set.max_events=100 data.val_set.max_events=50 fit.num_epochs=1 device=cpu
  ```
- **NEVER CANCEL** training runs - even short test runs can take 30+ seconds per epoch.

### Running the Application
- **Training**: `./train.py` or `python train.py`
  - Uses Hydra configuration from `config/config.yaml`
  - Default config trains transformer model on inner track selection task
  - Logs are saved to `logs/` directory (ignored in git)
  - Checkpoints saved to `logs/{exp_name}/{run_id}/checkpoints/`
- **GPU vs CPU**: 
  - Default config uses `device: cuda:0`
  - For CPU testing: add `device=cpu` to command line
  - GPU version requires CUDA-compatible PyTorch installation

### Experiment Tracking
- **Initialize Aim**: `aim init` (one-time setup)
- **Start Aim UI**: `aim up --port <PORT>` 
- **Access UI**: Navigate to `http://localhost:<PORT>`
- For remote servers, set up port forwarding: `ssh -L <PORT>:localhost:<PORT> user@server`

## Validation Scenarios

After making changes, ALWAYS validate by running these specific scenarios:

### 1. Basic Functionality Test
```bash
# Takes ~15 seconds - validates environment and data loading
./train.py data.train_set.path=data/sanity-check.root data.val_set.path=data/sanity-check.root data.test_set.path=data/sanity-check.root data.train_set.max_events=100 data.val_set.max_events=50 fit.num_epochs=1 device=cpu
```

### 2. Configuration Override Test
```bash
# Tests Hydra configuration system - should run with different model settings
./train.py model.module.dim_model=32 model.module.num_layers=2 fit.num_epochs=1 data.train_set.max_events=50 device=cpu
```

### 3. Multi-Epoch Training Test  
```bash
# Tests training loop and checkpointing - takes ~1-2 minutes
./train.py data.train_set.path=data/sanity-check.root data.val_set.path=data/sanity-check.root fit.num_epochs=5 data.train_set.max_events=500 device=cpu
```

## Timing Expectations

- **Environment creation**: 15-20 minutes (NEVER CANCEL - set 30+ minute timeout)
- **Quick sanity check** (1 epoch, 100 samples): ~15 seconds
- **Multi-epoch test** (5 epochs, 500 samples): ~1-2 minutes  
- **Full training** (100 epochs): Several hours (depends on data size)
- **NEVER CANCEL** any training command - always wait for completion or set appropriate long timeouts

## Key Configuration Files

### Repository Structure
```
/home/runner/work/DeepMuonReco/DeepMuonReco/
├── README.md                 # Basic setup instructions
├── environment.yaml          # Conda environment specification
├── setup.sh / setup.fish     # Environment activation scripts
├── train.py                  # Main training script
├── data/
│   └── sanity-check.root     # Sample data for testing
├── config/                   # Hydra configuration files
│   ├── config.yaml          # Main configuration
│   ├── model/transformer.yaml
│   ├── data/default.yaml
│   └── ...
└── src/deepmuonreco/        # Python package source
    ├── nn/models/           # Neural network models
    ├── data/datasets/       # Dataset classes
    └── ...
```

### Key Config Files Content
- `config/config.yaml`: Main configuration with defaults
- `config/data/default.yaml`: Dataset paths and data loader settings
- `config/model/transformer.yaml`: Model architecture configuration
- `environment.yaml`: Python dependencies (pytorch, aim, hydra, uproot, etc.)

## Common Issues and Solutions

### Environment Issues
- **"micromamba: command not found"**: Use conda instead of micromamba
- **SSL certificate errors with micromamba**: Switch to conda for environment creation
- **"Environment variable 'PROJECT_ROOT' not found"**: Set `export PROJECT_ROOT=$(pwd)` before running

### Training Issues  
- **CUDA not available warnings**: Expected when running on CPU, safe to ignore
- **"Aim repository not found"**: Run `aim init` first
- **Config interpolation errors**: Ensure all required environment variables are set

### Data Path Issues
- Default data paths in config point to `/users/seyang/data/deepmuonreco/`
- For testing, override with: `data.train_set.path=data/sanity-check.root`
- Sample data is available in `data/sanity-check.root`

## Development Workflow

1. **Always activate environment first** using setup scripts or manual activation
2. **Set required environment variables** (`PROJECT_ROOT`, `PYTHONPATH`)  
3. **Test changes with quick sanity check** before longer runs
4. **Use configuration overrides** to test different settings without modifying files
5. **Initialize Aim tracking** if working with experiment logging
6. **NEVER CANCEL long-running operations** - builds and training can take significant time

## Dependencies and Requirements

- **Python**: 3.12
- **Core ML**: PyTorch 2.6.0, TensorDict, Torchmetrics
- **Configuration**: Hydra 1.3.2 with colorlog extension
- **Experiment Tracking**: Aim
- **Data Processing**: Uproot (ROOT file handling), Pandas, NumPy
- **Visualization**: Matplotlib, Hist
- **Development**: No linting/formatting tools configured

The project is designed for high-energy physics applications and specifically handles ROOT files containing muon detector data.