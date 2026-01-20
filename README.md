# DREAM-RNN Training for lentiMPRA

A standalone implementation for training DREAM-RNN models on lentiMPRA data, based on the DREAM paper architecture.

## Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd dream_rnn_lentimpra
```

### 2. Set Up Environment

**Using uv (recommended - faster):**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
uv pip install h5py numpy pandas scipy scikit-learn pyyaml tqdm
```

**Using pip (alternative):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install h5py numpy pandas scipy scikit-learn pyyaml tqdm
```

**Note**: PyTorch installation is handled automatically by the training script. The script will:
- Detect your CUDA driver version
- Install a compatible PyTorch build automatically (using `uv` if available, otherwise `pip`)
- Verify GPU availability before training

If you prefer to install PyTorch manually, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

### 3. Download Data

The lentiMPRA dataset is available on Zenodo:

```bash
mkdir -p data
cd data
curl -L 'https://zenodo.org/record/14145285/files/data.tar.gz?download=1' -o data.tar.gz
tar -xzf data.tar.gz
mv data/* .
rm -rf data data.tar.gz
cd ..
```

### 4. Quick Test (1% data, ~2-3 min)

```bash
python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/test_run \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 0.01 \
    --ix 0 \
    --gpu 0
```

### 5. Full Training

```bash
python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/training \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 1.0 \
    --ix 0 \
    --gpu 0
```

**Training Time**: ~2.5-4 hours on GPU (CPU training is not supported)

### 6. Evaluate a Trained Model on the Test Set

```bash
python evaluate_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --model results/training/0_model.pth \
    --out results/eval_0 \
    --downsample 1.0 \
    --ix 0 \
    --gpu 0
```

This will:
- Load the saved model
- Evaluate on the held-out test set
- Save metrics (`performance.csv`) and predictions/targets (`*.npy`) in `results/eval_0/`.

### 7. Check Environment (Optional but Recommended)

Before long training runs, you can sanity-check your setup:

```bash
python check_env.py --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5
```

This tests:
- Python version
- PyTorch & CUDA availability
- GPU devices
- Data file presence and basic HDF5 structure

## Requirements

- Python 3.8+
- **CUDA-capable GPU (required)** – the scripts will error if no GPU is available
- NVIDIA drivers installed
- PyTorch (will be installed automatically with compatible CUDA version)
- h5py, numpy, pandas, scipy, scikit-learn, pyyaml, tqdm

**Automatic CUDA Detection**: The training and evaluation scripts automatically:
- Detect your CUDA driver version
- Install a compatible PyTorch build if needed
- Verify GPU availability before running

See `requirements.txt` for package versions.

## Output

After training, results are saved to the output directory:
- `{ix}_model.pth` - Trained model weights
- `{ix}_predictions.npy` - Test predictions
- `{ix}_targets.npy` - Test targets
- `{ix}_performance.csv` - Performance metrics (Pearson, Spearman, MSE)

## Training on Remote Servers

For training on remote GPU servers:

1. **Transfer repository** to server (via git clone or scp)
2. **Set up environment** on server (same as steps 2-3 above)
3. **Run training** with background execution:
   ```bash
   nohup python train_DREAM_RNN_lentiMPRA.py \
       --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
       --out results/training \
       --config DREAM_RNN_lentiMPRA.yaml \
       --downsample 1.0 \
       --ix 0 \
       --gpu 0 > training.log 2>&1 &
   ```
4. **Monitor progress**: `tail -f training.log`

## Ensemble Training

Train multiple models with different initializations:

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

## Configuration

Modify `DREAM_RNN_lentiMPRA.yaml` to adjust:
- `epochs`: Number of training epochs (default: 80)
- `batch_size`: Batch size (default: 1024)
- `optim_lr`: Learning rate (default: 0.005)

## Documentation

- **TRAINING_GUIDE.md** - Complete training guide with troubleshooting

## Data Source

Pre-processed lentiMPRA data is available on Zenodo:
- **DOI**: 10.5281/zenodo.14145284
- **Link**: https://zenodo.org/records/14145285

## Citation

If you use this code, please cite the original papers:
- DREAM paper (for the architecture)
- DEGU paper (for the ensemble/distillation methodology)

See the original repository for full citations: https://github.com/zrcjessica/ensemble_distillation

## License

MIT License
