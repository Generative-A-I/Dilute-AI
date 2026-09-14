from __future__ import annotations

import torch


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def precision_flags(device: torch.device, precision: str) -> tuple[bool, bool]:
    if precision == "no":
        return False, False
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested, but this CUDA GPU does not support it.")
        return True, False
    if precision == "fp16":
        if device.type != "cuda":
            raise RuntimeError("fp16 training is supported only on CUDA in this project.")
        return False, True
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return True, False
    return False, device.type == "cuda"
