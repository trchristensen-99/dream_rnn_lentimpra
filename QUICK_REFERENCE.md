# Quick Reference

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Download Data

```bash
mkdir -p data
cd data
curl -L 'https://zenodo.org/record/14145285/files/data.tar.gz?download=1' -o data.tar.gz
tar -xzf data.tar.gz && mv data/* . && rm -rf data data.tar.gz
cd ..
```

## Training Commands

**Quick test (1% data):**
```bash
python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/test \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 0.01 --ix 0 --gpu 0
```

**Full training:**
```bash
python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/training \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 1.0 --ix 0 --gpu 0
```

**Background execution (remote servers):**
```bash
nohup python train_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --out results/training \
    --config DREAM_RNN_lentiMPRA.yaml \
    --downsample 1.0 --ix 0 --gpu 0 > training.log 2>&1 &
```

## Output Files

- `{ix}_model.pth` - Model weights
- `{ix}_predictions.npy` - Test predictions
- `{ix}_targets.npy` - Test targets
- `{ix}_performance.csv` - Metrics

For detailed instructions, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
