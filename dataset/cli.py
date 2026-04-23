"""Dataset generation CLI adapter."""

import logging

from utils.config import Config
from dataset.generator import generate_dataset
from dataset.writer import save, validate


def run(config: Config, logger: logging.Logger) -> int:
    """Run dataset generation command.

    Args:
        config: Configuration object.
        logger: Logger instance.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        logger.info(f"Generating dataset of size {config.dataset_size}")
        dataset = generate_dataset(config.dataset_size)

        logger.info(f"Saving to {config.dataset_path}")
        save(dataset, config.dataset_path)

        logger.info("Validating dataset")
        stats = validate(config.dataset_path)

        logger.info(
            f"Dataset saved and validated: "
            f"{stats.total_count} total, "
            f"{stats.math_count} math, "
            f"{stats.generic_count} generic, "
            f"{stats.malformed_count} malformed, "
            f"avg length: {stats.avg_char_length:.0f}"
        )

        return 0

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        return 1
