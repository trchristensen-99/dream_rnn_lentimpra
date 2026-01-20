# Complete Guide: Training DREAM-RNN on lentiMPRA Data

This guide provides step-by-step instructions for training DREAM-RNN models on the lentiMPRA dataset.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Setup](#repository-setup)
3. [Data Download](#data-download)
4. [Environment Setup](#environment-setup)
5. [Training the Model](#training-the-model)
6. [Training on Remote Servers](#training-on-remote-servers)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.8 or higher
- **CUDA-capable GPU (required)** – training and evaluation scripts require a GPU
- ~2GB disk space for data
- ~5GB disk space for models and outputs

### Required Python Packages

- PyTorch (with CUDA support if using GPU)
- h5py
- numpy
- pandas
- scipy
- scikit-learn
- pyyaml
- tqdm

---

## Repository Setup

```bash
git clone <your-repo-url>
cd dream_rnn_lentimpra
```

---

## Data Download

The lentiMPRA dataset is available on Zenodo:

```bash
# Create data directory
mkdir -p data

# Download the data archive (~1.7GB)
cd data
curl -L 'https://zenodo.org/record/14145285/files/data.tar.gz?download=1' -o data.tar.gz

# Extract the archive
tar -xzf data.tar.gz

# Move files to the correct location
mv data/* .
rm -rf data data.tar.gz
cd ..
```

### Verify Data

```bash
python3 -c "
import h5py
f = h5py.File('data/lentiMPRA_K562_activity_and_aleatoric_data.h5', 'r')
print('Keys:', list(f.keys()))
print('Train/X shape:', f['Train/X'].shape)
print('Train/y shape:', f['Train/y'].shape)
f.close()
"
```

Expected output:
- Train/X: (180564, 230, 4)
- Train/y: (180564, 2)

---

## Environment Setup

### 1. Create Virtual Environment

**Using uv (recommended - faster):**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using pip (alternative):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

**Automatic Installation (Recommended):**

The training script will automatically detect your CUDA driver version and install a compatible PyTorch build. You only need to install the other dependencies:

**With uv:**
```bash
uv pip install h5py numpy pandas scipy scikit-learn pyyaml tqdm
```

**With pip:**
```bash
pip install --upgrade pip
pip install h5py numpy pandas scipy scikit-learn pyyaml tqdm
```

The script will automatically use `uv` if available, otherwise fall back to `pip` for PyTorch installation.

**Manual Installation (Optional):**

If you prefer to install PyTorch manually, you can do so based on your CUDA version:

**With uv:**
```bash
# For CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**With pip:**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

The script will verify compatibility and install the correct version if needed.

### 3. Verify Installation

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"
```

### 4. Optional: Environment Check Script

Run the provided check script to validate your setup:

```bash
python check_env.py --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5
```

This will report:
- Python version
- PyTorch version and CUDA availability
- Detected GPUs
- Basic checks on the lentiMPRA HDF5 data file

---

## Training the Model

### Quick Test (1% of data, ~2-3 minutes)

```bash
python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/test_run \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 0.01 \
    --ix 0 \
    --gpu 0
```

### Full Training (Complete dataset)

```bash
python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/training \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 1.0 \
    --ix 0 \
    --gpu 0
```

**Training Time Estimates:**
- **GPU (A4000/A3000)**: 2.5-4 hours
- **CPU**: 10-20 hours

### Training Multiple Models (Ensemble)

```bash
for i in {0..9}; do
    python train_DREAM_RNN_lentiMPRA.py \
        --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
        --out results/ensemble \
        --config DREAM_RNN_lentiMPRA.yaml \
        --downsample 1.0 \
        --ix $i \
        --gpu 0
done
```

---

## Training on Remote Servers

For training on remote GPU servers, you can use background execution to keep training running after disconnecting:

### Using nohup (Recommended)

```bash
# Run training in background with logging
nohup python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/training \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 1.0 \
    --ix 0 \
    --gpu 0 > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check GPU usage (if available)
nvidia-smi
```

### Using screen (For Persistent Sessions)

```bash
# Start a screen session
screen -S dream_training

# In the screen session, run training
python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/training \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 1.0 \
    --ix 0 \
    --gpu 0

# Detach: Press Ctrl+A then D
# Reattach later: screen -r dream_training
```

### Using tmux (Alternative)

```bash
# Start a tmux session
tmux new -s dream_training

# Run training in the session
python train_DREAM_RNN_lentiMPRA.py ...

# Detach: Press Ctrl+B then D
# Reattach: tmux attach -t dream_training
```

---

## Output Files

After training completes, the following files will be created in the output directory:

- `{ix}_model.pth` - Trained model weights
- `{ix}_predictions.npy` - Test set predictions
- `{ix}_targets.npy` - Test set ground truth
- `{ix}_performance.csv` - Performance metrics (Pearson, Spearman, MSE)

### Evaluation Script (Held-out Test Set)

You can also evaluate a saved model checkpoint using the standalone evaluation script:

```bash
python evaluate_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --model results/training/0_model.pth \
    --out results/eval_0 \
    --downsample 1.0 \
    --ix 0 \
    --gpu 0
```

This script:
- Loads the lentiMPRA data and a trained model
- Evaluates on the test split
- Writes:
  - `performance.csv` (metrics)
  - `predictions.npy` (test predictions)
  - `targets.npy` (test targets)

### Performance Metrics

The `*_performance.csv` file contains:
- `activity_pearson` - Pearson correlation for activity predictions
- `activity_spearman` - Spearman correlation for activity predictions
- `activity_mse` - Mean squared error for activity
- `aleatoric_pearson` - Pearson correlation for aleatoric uncertainty
- `aleatoric_spearman` - Spearman correlation for aleatoric uncertainty
- `aleatoric_mse` - Mean squared error for aleatoric uncertainty
- `avg_pearson`, `avg_spearman`, `avg_mse` - Average metrics

---

## Troubleshooting

### PyTorch / CUDA Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'torch._prims_common'`

**Solution**: Reinstall PyTorch:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CUDA Driver Version Mismatch

**Problem**: `CUDA initialization: The NVIDIA driver on your system is too old`

**Solution**: Either:
1. Update your NVIDIA driver, or
2. Install a PyTorch version compatible with your CUDA driver version

Check your CUDA version:
```bash
nvidia-smi
```

### Data File Not Found

**Problem**: `FileNotFoundError: data/lentiMPRA_K562_activity_and_aleatoric_data.h5`

**Solution**: 
1. Verify the file exists: `ls -lh data/lentiMPRA_K562_activity_and_aleatoric_data.h5`
2. Download the data following the [Data Download](#data-download) section
3. Check you're running from the repository root directory

### Out of Memory Errors

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
1. Reduce batch size in `DREAM_RNN_lentiMPRA.yaml`:
   - Change `batch_size: 1024` to `batch_size: 512` or lower
2. Use a smaller downsample ratio: `--downsample 0.1`
3. Use a different GPU with more memory

### Training is Slow or Falls Back to CPU

**Problem**: Training is extremely slow, or logs show `CUDA available: False`

**Solutions**:
1. Verify CUDA is available: ensure `CUDA available: True` is printed at startup
2. If not, fix your CUDA/PyTorch installation (see issues above) – **CPU-only training is not supported by this script**
3. Check GPU utilization: `nvidia-smi` should show usage during training

---

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to HDF5 data file | Required |
| `--out` | Output directory for models and results | Required |
| `--config` | Path to YAML config file | `DREAM_RNN_lentiMPRA.yaml` |
| `--downsample` | Fraction of training data to use (0.0-1.0) | 1.0 |
| `--ix` | Model index (for ensemble training) | 0 |
| `--gpu` | GPU device ID | 0 |

---

## Configuration

The training configuration is in `DREAM_RNN_lentiMPRA.yaml`:

- `epochs`: Number of training epochs (default: 80)
- `batch_size`: Batch size (default: 1024)
- `optim_lr`: Learning rate (default: 0.005)
- `optimizer`: Optimizer type (default: AdamW)
- `aleatoric`: Whether to predict aleatoric uncertainty (default: false)
- `epistemic`: Whether to predict epistemic uncertainty (default: false)

Modify these values as needed for your experiments.

---

## Additional Resources

- **Data**: https://zenodo.org/records/14145285
- **Original Repository**: https://github.com/zrcjessica/ensemble_distillation
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
