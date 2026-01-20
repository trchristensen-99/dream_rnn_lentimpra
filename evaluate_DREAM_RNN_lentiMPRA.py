#!/usr/bin/env python3
"""
Evaluate a trained DREAM-RNN model on the lentiMPRA test set.

This script:
- Loads the lentiMPRA HDF5 data file
- Loads a saved DREAM-RNN model checkpoint
- Computes performance metrics on the held-out test set
- Saves metrics and predictions to the output directory

Usage example:

python evaluate_DREAM_RNN_lentiMPRA.py \
    --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5 \
    --model results/training/0_model.pth \
    --out results/eval_0 \
    --config DREAM_RNN_lentiMPRA.yaml \
    --gpu 0
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import importlib

from train_DREAM_RNN_lentiMPRA import (
    DREAM_RNN_LentiMPRA,
    load_lentiMPRA_data,
    evaluate_model,
)
from cuda_utils import check_and_fix_cuda_compatibility


def main(args: argparse.Namespace) -> None:
    print("=== Starting DREAM-RNN Evaluation ===")
    
    # Check and fix CUDA compatibility
    print("\n=== Checking CUDA Compatibility ===")
    if not check_and_fix_cuda_compatibility():
        raise RuntimeError(
            "CUDA GPU is required for evaluation but was not detected or is incompatible. "
            "The script attempted to fix compatibility issues automatically. "
            "If problems persist, please check your NVIDIA driver and PyTorch installation."
        )
    
    # Re-import torch after potential reinstall
    import torch
    importlib.reload(torch)
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for evaluation but was not detected after compatibility check. "
            "Please ensure NVIDIA drivers are installed and PyTorch is built with CUDA support."
        )
    
    # Set device (GPU only)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"\n✓ CUDA is available")
    print(f"Using device: {device}")

    # Load data (we only need the test set, but reuse loader)
    print(f"Loading data from {args.data}")
    X_train, y_train, X_test, y_test, X_val, y_val = load_lentiMPRA_data(args)
    del X_train, y_train, X_val, y_val  # Free memory, keep only test

    # Create test loader
    from torch.utils.data import DataLoader, TensorDataset

    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model and load weights
    model = DREAM_RNN_LentiMPRA(in_channels=4, seqsize=230).to(device)
    model_path = Path(args.model)
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Evaluate
    print("Evaluating model on test set...")
    metrics, predictions, targets = evaluate_model(model, test_loader, device)

    print("\nTest Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save results
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_dir / "performance.csv", index=False)
    np.save(out_dir / "predictions.npy", predictions)
    np.save(out_dir / "targets.npy", targets)

    print(f"\nSaved evaluation results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DREAM-RNN model on lentiMPRA test set"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to lentiMPRA HDF5 data file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model .pth file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="DREAM_RNN_lentiMPRA.yaml",
        help="Config file (kept for compatibility with training script)",
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Downsample ratio (must match training if you used downsampling)",
    )
    parser.add_argument(
        "--ix",
        type=int,
        default=0,
        help="Model index (used for seeding; should match training)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for evaluation",
    )

    main(parser.parse_args())

