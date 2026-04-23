"""Device detection and management for LLM fine-tuning."""

import os
from typing import Any, Dict, Optional


def _try_import_torch() -> Optional[object]:
    """Best-effort torch import.

    Returns:
        The imported torch module, or None if torch isn't installed or fails to load.
    """
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        # On Windows, PyTorch can be installed but still fail to import with
        # WinError 126 (missing DLL dependency). Treat it as unavailable.
        return None


def detect_device() -> str:
    """Detect the best available compute device.

    Returns:
        One of "cuda", "mps", or "cpu".
    """
    # Check for explicit DEVICE env var override
    explicit_device = os.getenv("DEVICE")
    if explicit_device:
        return explicit_device

    torch = _try_import_torch()
    if torch is None:
        return "cpu"

    # Check CUDA availability
    if torch.cuda.is_available():
        return "cuda"

    # Check MPS availability (Mac with Metal Performance Shaders)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return "mps"

    # Fall back to CPU
    return "cpu"


def device_defaults(device: str) -> Dict[str, Any]:
    """Get device-specific defaults.

    Args:
        device: Device type ("cuda", "mps", or "cpu").

    Returns:
        Dictionary with batch_size and fp16 defaults.
    """
    if device == "cuda":
        return {
            "batch_size": 4,
            "fp16": True,
        }
    elif device == "mps":
        return {
            "batch_size": 1,
            "fp16": False,
        }
    else:  # cpu
        return {
            "batch_size": 1,
            "fp16": False,
        }


def to_device(model: Any, device: str) -> Any:
    """Move model to device.

    Args:
        model: Model to move.
        device: Device type ("cuda", "mps", or "cpu").

    Returns:
        Model on the target device.
    """
    return model.to(device)
