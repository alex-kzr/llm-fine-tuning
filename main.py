"""CLI entrypoint for LLM fine-tuning pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from utils.config import load_config, Config
from utils.logging import configure_root_logger


def _apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI argument overrides to config."""
    if hasattr(args, 'epochs') and args.epochs is not None:
        config.epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.batch_size = args.batch_size
    if hasattr(args, 'dataset') and args.dataset is not None:
        config.dataset_path = args.dataset
    if hasattr(args, 'size') and args.size is not None:
        config.dataset_size = args.size
    if hasattr(args, 'adapter') and args.adapter is not None:
        config.adapter_output = args.adapter
    if hasattr(args, 'output') and args.output is not None:
        config.merged_output = args.output
    if hasattr(args, 'questions_file') and args.questions_file is not None:
        config.questions_path = args.questions_file
    return config


def generate_dataset(config: Config, logger: logging.Logger) -> int:
    """Generate dataset command."""
    try:
        from dataset import cli as dataset_cli
        return dataset_cli.run(config, logger)
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        return 1


def train(config: Config, logger: logging.Logger) -> int:
    """Train command."""
    try:
        from training import cli as training_cli
        return training_cli.run(config, logger)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def test(config: Config, logger: logging.Logger) -> int:
    """Test/inference command."""
    try:
        from inference import cli as inference_cli
        return inference_cli.run(config, logger)
    except Exception as e:
        logger.error(f"Inference/testing failed: {e}", exc_info=True)
        return 1


def merge(config: Config, logger: logging.Logger) -> int:
    """Merge LoRA adapter command."""
    try:
        from training import cli as training_cli
        return training_cli.merge(config, logger)
    except Exception as e:
        logger.error(f"Merging failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="LLM fine-tuning pipeline with LoRA"
    )

    parser.add_argument(
        "--env-file",
        "--config",
        dest="env_file",
        type=str,
        help="Path to an extra env file (default .env is always loaded first)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-dataset command
    gen_parser = subparsers.add_parser("generate-dataset", help="Generate dataset")
    gen_parser.add_argument("--size", type=int, help="Dataset size")
    gen_parser.set_defaults(func=generate_dataset)

    # train command
    train_parser = subparsers.add_parser("train", help="Train model with LoRA")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--dataset", type=str, help="Dataset file path")
    train_parser.set_defaults(func=train)

    # test command
    test_parser = subparsers.add_parser("test", help="Test/run inference")
    test_parser.add_argument("--adapter", type=str, help="Path to LoRA adapter")
    test_parser.add_argument("--questions-file", type=str, help="Questions file path")
    test_parser.set_defaults(func=test)

    # merge command
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter to model")
    merge_parser.add_argument("--adapter", type=str, help="Path to LoRA adapter")
    merge_parser.add_argument("--output", type=str, help="Output model path")
    merge_parser.set_defaults(func=merge)

    args = parser.parse_args()

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 0

    # Load config and logger
    config = load_config(args.env_file)
    config = _apply_cli_overrides(config, args)
    logger = configure_root_logger(verbose=args.verbose, command=args.command)

    logger.info(f"Starting command: {args.command}")
    logger.info(f"Detected device: {config.device}")
    try:
        import torch

        if config.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "DEVICE=cuda requested but torch.cuda.is_available() is False. "
                "This usually means the installed PyTorch build has no CUDA support "
                "(e.g. '+cpu'), or NVIDIA drivers are missing/incompatible."
            )
        if config.device == "cpu" and torch.version.cuda is None:
            logger.info(
                "PyTorch appears to be installed as CPU-only (torch.version.cuda is None). "
                "Install a CUDA-enabled PyTorch wheel to use GPU acceleration."
            )
    except Exception:
        # Best-effort diagnostics only; training/inference modules will import torch anyway.
        pass
    logger.debug(f"Config: {config}")

    # Execute command
    try:
        result = args.func(config, logger)
        return result if isinstance(result, int) else 0
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
