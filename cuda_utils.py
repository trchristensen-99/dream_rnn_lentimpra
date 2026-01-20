#!/usr/bin/env python3
"""
CUDA version detection and PyTorch compatibility utilities.

This module automatically detects the CUDA driver version and ensures
PyTorch is installed with a compatible CUDA build.
"""

import subprocess
import sys
import re
from typing import Optional, Tuple


def get_cuda_driver_version() -> Optional[str]:
    """
    Detect CUDA driver version from nvidia-smi.
    
    Returns:
        CUDA version string (e.g., "11.4", "12.1") or None if nvidia-smi fails
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Get CUDA version from nvidia-smi (different command)
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Look for "CUDA Version: X.Y" in output
                match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
                if match:
                    return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def get_pytorch_cuda_version() -> Optional[str]:
    """
    Get the CUDA version that PyTorch was built for.
    
    Returns:
        CUDA version string (e.g., "11.8", "12.1") or None if PyTorch not installed
    """
    try:
        import torch
        if hasattr(torch.version, "cuda") and torch.version.cuda:
            return torch.version.cuda
    except ImportError:
        pass
    return None


def get_pytorch_index_url(cuda_version: str) -> Optional[str]:
    """
    Get the PyTorch index URL for a given CUDA version.
    
    Args:
        cuda_version: CUDA version string (e.g., "11.4", "12.1")
    
    Returns:
        Index URL string or None if version not supported
    """
    # Map CUDA versions to PyTorch index URLs
    version_map = {
        "12.1": "https://download.pytorch.org/whl/cu121",
        "12.4": "https://download.pytorch.org/whl/cu124",
        "11.8": "https://download.pytorch.org/whl/cu118",
        "11.7": "https://download.pytorch.org/whl/cu117",
    }
    
    # Try exact match first
    if cuda_version in version_map:
        return version_map[cuda_version]
    
    # Try major.minor match (e.g., 11.4 -> 11.8, 12.0 -> 12.1)
    major_minor = cuda_version.rsplit(".", 1)[0]
    for version, url in version_map.items():
        if version.startswith(major_minor + "."):
            return url
    
    # Default to latest CUDA 12.1 if no match
    return version_map.get("12.1")


def check_and_fix_cuda_compatibility() -> bool:
    """
    Check CUDA compatibility and attempt to fix if needed.
    
    Returns:
        True if CUDA is available and compatible, False otherwise
    """
    # Get CUDA driver version
    driver_version = get_cuda_driver_version()
    if driver_version is None:
        print("⚠ Warning: Could not detect CUDA driver version from nvidia-smi")
        print("  Make sure nvidia-smi is available and NVIDIA drivers are installed")
    
    # Check if PyTorch is installed
    try:
        import torch
        pytorch_cuda = get_pytorch_cuda_version()
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  Driver version: {driver_version or 'unknown'}")
            print(f"  PyTorch CUDA: {pytorch_cuda}")
            if driver_version and pytorch_cuda:
                # Check if versions are compatible (same major.minor)
                driver_major_minor = ".".join(driver_version.split(".")[:2])
                pytorch_major_minor = ".".join(pytorch_cuda.split(".")[:2])
                if driver_major_minor != pytorch_major_minor:
                    print(f"  ⚠ Version mismatch: driver {driver_version} vs PyTorch {pytorch_cuda}")
                    print(f"    This may work, but consider matching versions for best performance")
            return True
        else:
            # CUDA not available - check if it's a version mismatch
            if driver_version and pytorch_cuda:
                driver_major_minor = ".".join(driver_version.split(".")[:2])
                pytorch_major_minor = ".".join(pytorch_cuda.split(".")[:2])
                
                if driver_major_minor != pytorch_major_minor:
                    print(f"✗ CUDA version mismatch detected:")
                    print(f"  Driver version: {driver_version}")
                    print(f"  PyTorch built for: {pytorch_cuda}")
                    print(f"\n  Attempting to install compatible PyTorch...")
                    
                    # Try to install compatible version
                    index_url = get_pytorch_index_url(driver_version)
                    if index_url:
                        print(f"  Installing PyTorch for CUDA {driver_version}...")
                        try:
                            # Try uv first, fall back to pip if not available
                            try:
                                subprocess.run(["uv", "--version"], check=True, capture_output=True)
                                use_uv = True
                            except (subprocess.CalledProcessError, FileNotFoundError):
                                use_uv = False
                            
                            if use_uv:
                                subprocess.run(
                                    ["uv", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
                                    check=True,
                                    capture_output=True,
                                )
                                subprocess.run(
                                    [
                                        "uv", "pip", "install",
                                        "torch", "torchvision", "torchaudio",
                                        "--index-url", index_url,
                                    ],
                                    check=True,
                                )
                            else:
                                subprocess.run(
                                    [
                                        sys.executable,
                                        "-m",
                                        "pip",
                                        "uninstall",
                                        "-y",
                                        "torch",
                                        "torchvision",
                                        "torchaudio",
                                    ],
                                    check=True,
                                    capture_output=True,
                                )
                                subprocess.run(
                                    [
                                        sys.executable,
                                        "-m",
                                        "pip",
                                        "install",
                                        "torch",
                                        "torchvision",
                                        "torchaudio",
                                        "--index-url",
                                        index_url,
                                    ],
                                    check=True,
                                )
                            print(f"  ✓ PyTorch reinstalled for CUDA {driver_version}")
                            print(f"  ⚠ Please restart the script for changes to take effect")
                            print(f"    The new PyTorch installation requires a fresh Python process")
                            return False  # Return False so user restarts
                        except subprocess.CalledProcessError as e:
                            print(f"  ✗ Failed to reinstall PyTorch: {e}")
                            print(f"    Please manually install PyTorch for CUDA {driver_version}:")
                            print(f"    pip install torch torchvision torchaudio --index-url {index_url}")
                            return False
                    else:
                        print(f"  ✗ No PyTorch build available for CUDA {driver_version}")
                        print(f"    Please update your NVIDIA driver or install a compatible PyTorch")
                        return False
                else:
                    print(f"✗ CUDA not available despite version match")
                    print(f"  Driver: {driver_version}, PyTorch: {pytorch_cuda}")
                    print(f"  This may indicate a driver or hardware issue")
                    return False
            else:
                print(f"✗ CUDA not available")
                if not driver_version:
                    print(f"  Could not detect CUDA driver - make sure NVIDIA drivers are installed")
                if not pytorch_cuda:
                    print(f"  PyTorch was not built with CUDA support")
                    print(f"  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                return False
    except ImportError:
        print("✗ PyTorch is not installed")
        if driver_version:
            index_url = get_pytorch_index_url(driver_version)
            if index_url:
                print(f"  Installing PyTorch for CUDA {driver_version}...")
                try:
                    # Try uv first, fall back to pip if not available
                    try:
                        subprocess.run(["uv", "--version"], check=True, capture_output=True)
                        use_uv = True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        use_uv = False
                    
                    if use_uv:
                        subprocess.run(
                            [
                                "uv", "pip", "install",
                                "torch", "torchvision", "torchaudio",
                                "--index-url", index_url,
                            ],
                            check=True,
                        )
                    else:
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                "torch",
                                "torchvision",
                                "torchaudio",
                                "--index-url",
                                index_url,
                            ],
                            check=True,
                        )
                    print(f"  ✓ PyTorch installed")
                    # Re-check
                    import torch
                    if torch.cuda.is_available():
                        return True
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Failed to install PyTorch: {e}")
        return False
    
    return False
