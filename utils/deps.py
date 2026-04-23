"""Helpers for reporting missing/broken optional dependencies."""

from __future__ import annotations

import logging


def log_dependency_import_error(
    *,
    logger: logging.Logger,
    command_name: str,
    required_hint: str,
    exc: BaseException,
) -> None:
    """Log a helpful message when optional ML deps fail to import.

    This project uses heavyweight optional dependencies (torch/transformers/etc).
    In particular, on Windows PyTorch can be installed but fail to import due to
    missing native DLL dependencies (commonly WinError 126).
    """
    if (
        isinstance(exc, OSError)
        and getattr(exc, "winerror", None) == 126
        and ("torch_python.dll" in str(exc) or "torch\\lib\\" in str(exc))
    ):
        logger.error(
            "PyTorch is installed but failed to load native DLLs on Windows (WinError 126).",
            exc_info=True,
        )
        logger.error(
            "Fix: install Microsoft Visual C++ Redistributable 2015–2022 (x64), "
            "then reinstall a compatible PyTorch wheel."
        )
        logger.error("PowerShell (CPU-only PyTorch):")
        logger.error("  pip uninstall -y torch torchvision torchaudio")
        logger.error(
            "  pip install --upgrade --force-reinstall torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        logger.error(
            "If you need CUDA, install the CUDA wheel from the official selector on pytorch.org "
            "instead of the CPU index."
        )
        return

    logger.error(
        f"Missing {command_name} dependencies. {required_hint}",
        exc_info=True,
    )

