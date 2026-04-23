"""Configuration management for LLM fine-tuning."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from utils.device import detect_device


@dataclass
class Config:
    """Configuration for LLM fine-tuning pipeline."""

    hf_token: Optional[str] = None
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_size: int = 300
    dataset_path: str = "dataset.jsonl"
    questions_path: str = "test_dataset.jsonl"
    adapter_output: str = "./tinyllama-json"
    merged_output: str = "./tinyllama-merged"
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 2
    learning_rate: float = 2e-4
    max_length: int = 512
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        # Override from environment variables if present
        self.hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or self.hf_token
        )
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_token)

        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.dataset_size = int(os.getenv("DATASET_SIZE", self.dataset_size))
        self.dataset_path = os.getenv("DATASET_PATH", self.dataset_path)
        self.questions_path = os.getenv("QUESTIONS_PATH", self.questions_path)
        self.adapter_output = os.getenv("ADAPTER_OUTPUT", self.adapter_output)
        self.merged_output = os.getenv("MERGED_OUTPUT", self.merged_output)
        self.batch_size = int(os.getenv("BATCH_SIZE", self.batch_size))
        self.gradient_accumulation_steps = int(
            os.getenv("GRADIENT_ACCUMULATION_STEPS", self.gradient_accumulation_steps)
        )
        self.epochs = int(os.getenv("EPOCHS", self.epochs))
        self.learning_rate = float(os.getenv("LEARNING_RATE", self.learning_rate))
        self.max_length = int(os.getenv("MAX_LENGTH", self.max_length))
        # Auto-detect device if not explicitly set
        if "DEVICE" in os.environ:
            self.device = os.getenv("DEVICE")
        else:
            self.device = detect_device()


def load_config(env_file: Optional[str] = None) -> Config:
    """Load configuration from environment variables and .env file.

    Args:
        env_file: Optional path to an additional env file.
            The default `.env` in the current directory is always loaded first (if present).

    Returns:
        Config object with loaded values.
    """
    default_env = Path(".env")
    if default_env.exists():
        load_dotenv(default_env)

    if env_file:
        env_path = Path(env_file)
        if env_path.exists() and env_path.resolve() != default_env.resolve():
            load_dotenv(env_path, override=True)

    return Config()
