#!/usr/bin/env python
"""Check whether the active MA-SP-SAM environment can run PyTorch on CUDA."""

from __future__ import annotations

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Torch CUDA device to test, e.g. cuda:0.")
    parser.add_argument("--size", type=int, default=256, help="Square matrix size for a tiny smoke operation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if not torch.cuda.is_available():
        print("cuda_available: False")
        print("reason: PyTorch cannot see CUDA devices in this process.")
        return 1

    print("cuda_available: True")
    print(f"device_count: {torch.cuda.device_count()}")
    device = torch.device(args.device)
    print(f"selected_device: {device}")
    print(f"device_name: {torch.cuda.get_device_name(device)}")

    x = torch.randn(args.size, args.size, device=device)
    y = x @ x.T
    torch.cuda.synchronize(device)
    print(f"smoke_tensor: shape={tuple(y.shape)} device={y.device} mean={float(y.mean()):.6f}")
    print(f"memory_allocated_mb: {torch.cuda.memory_allocated(device) / 1024 / 1024:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
