"""Structured logging configuration for LLM fine-tuning."""

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional

_loggers_configured = set()


def get_logger(name: str, verbose: bool = False, command: Optional[str] = None) -> logging.Logger:
    """Get or create a configured logger instance.

    Args:
        name: Logger name, typically __name__.
        verbose: Enable DEBUG level logging.
        command: Command name for log file naming (e.g., 'train', 'inference').

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated imports
    if name in _loggers_configured:
        return logger

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler.setLevel(console_level)

    # File handler (lazy-create logs/ directory)
    logs_dir = Path("./logs")
    if command or name == "root":
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        command_suffix = f"_{command}" if command else ""
        log_file = logs_dir / f"{timestamp}{command_suffix}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
    else:
        file_handler = None

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if file_handler:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers_configured.add(name)
    return logger


def configure_root_logger(verbose: bool = False, command: Optional[str] = None) -> logging.Logger:
    """Configure the root logger for CLI entry point.

    Args:
        verbose: Enable DEBUG level logging.
        command: Command name for log file naming.

    Returns:
        Configured root logger.
    """
    return get_logger("", verbose=verbose, command=command)
