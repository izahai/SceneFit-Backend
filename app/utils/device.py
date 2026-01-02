# app/utils/device.py

import torch

def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_dtype(
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.dtype:
    if dtype is not None:
        return dtype

    if device.type == "cuda":
        # Prefer bf16 if supported (A100/H100/RTX 40xx)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device.type == "mps":
        return torch.float16

    # CPU: always fp32 for safety
    return torch.float32
