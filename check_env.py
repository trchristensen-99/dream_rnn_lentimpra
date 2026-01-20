#!/usr/bin/env python3
"""
Environment check script for DREAM-RNN lentiMPRA training.

This script verifies:
- Python version
- PyTorch installation
- CUDA availability and GPU devices
- Presence of the lentiMPRA data file

Run this BEFORE training:

    python check_env.py --data data/lentiMPRA_K562_activity_and_aleatoric_data.h5
"""

import argparse
import os
import sys
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    print("=== DREAM-RNN lentiMPRA Environment Check ===\n")

    # Python
    print("Python:")
    print(f"  Executable: {sys.executable}")
    print(f"  Version   : {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("  ⚠ Python 3.8+ is recommended")
    else:
        print("  ✓ Python version is OK")
    print()

    # PyTorch / CUDA
    try:
        import torch  # type: ignore

        print("PyTorch:")
        print(f"  Version        : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available : {cuda_available}")
        if cuda_available:
            print(f"  CUDA version   : {torch.version.cuda}")
            device_count = torch.cuda.device_count()
            print(f"  GPU count      : {device_count}")
            for i in range(device_count):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print("  ✓ CUDA GPU detected and usable")
        else:
            print("  ✗ CUDA GPU not available - training script will raise an error")
        print()
    except ImportError as e:
        print("PyTorch:")
        print("  ✗ torch is not installed")
        print(f"  Error: {e}")
        print("  Install with (for CUDA 12.1):")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print()

    # Data file
    data_path = Path(args.data) if args.data else None
    print("Data file:")
    if data_path is None:
        print("  (no --data path provided; skipping)")
    else:
        print(f"  Expected path: {data_path}")
        if data_path.is_file():
            size_mb = data_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Found ({size_mb:.1f} MB)")
            try:
                import h5py  # type: ignore

                with h5py.File(data_path, "r") as f:
                    keys = list(f.keys())
                    print(f"  HDF5 keys     : {keys}")
                    if "Train" in keys and "X" in f["Train"]:
                        print(f"  Train/X shape : {f['Train']['X'].shape}")
            except ImportError:
                print("  ⚠ h5py not installed; cannot inspect contents")
            except Exception as e:
                print(f"  ⚠ Error reading HDF5 file: {e}")
        else:
            print("  ✗ Data file not found")
            print("  Make sure you've downloaded from Zenodo and placed it under data/")
    print()

    print("=== Check complete ===")
    print("If any items above are marked with ✗, fix them before running training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check environment for DREAM-RNN lentiMPRA training")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to lentiMPRA HDF5 data file (optional but recommended)",
    )
    main(parser.parse_args())

