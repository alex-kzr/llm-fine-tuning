"""Dataset writing and validation for LLM fine-tuning."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DatasetStats:
    """Statistics about a dataset."""

    total_count: int
    avg_char_length: float
    math_count: int
    generic_count: int
    malformed_count: int


def save(dataset: List[Dict[str, Any]], path: str) -> None:
    """Save dataset to JSONL file.

    Args:
        dataset: List of sample dictionaries with "text" key.
        path: Output file path.

    Raises:
        ValueError: If any sample is missing "text" key.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            if "text" not in sample:
                raise ValueError(f"Sample missing 'text' key: {sample}")
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")


def validate(path: str) -> DatasetStats:
    """Validate dataset file and return statistics.

    Args:
        path: Path to JSONL dataset file.

    Returns:
        DatasetStats with counts and statistics.

    Raises:
        ValueError: If validation fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"Dataset file not found: {path}")

    total_count = 0
    total_chars = 0
    math_count = 0
    generic_count = 0
    malformed_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse outer JSON
                sample = json.loads(line)

                if "text" not in sample:
                    malformed_count += 1
                    continue

                text = sample["text"]
                if not isinstance(text, str) or not text:
                    malformed_count += 1
                    continue

                total_count += 1
                total_chars += len(text)

                # Extract and validate embedded JSON from ### Response:
                response_match = re.search(
                    r"### Response:\s*(\{[^{}]*\"model\"[^{}]*\})",
                    text,
                    re.DOTALL
                )

                if response_match:
                    response_text = response_match.group(1)
                    try:
                        response = json.loads(response_text)

                        # Validate schema
                        required_keys = {"model", "type", "message"}
                        if not required_keys.issubset(response.keys()):
                            malformed_count += 1
                            continue

                        # Count math vs generic
                        if any(op in text for op in ["+", "*", "-", "/"]):
                            math_count += 1
                        else:
                            generic_count += 1

                    except json.JSONDecodeError:
                        malformed_count += 1
                else:
                    malformed_count += 1

            except json.JSONDecodeError as e:
                malformed_count += 1

    if total_count == 0 and malformed_count > 0:
        raise ValueError(f"All {malformed_count} samples in {path} are malformed")

    avg_length = total_chars / total_count if total_count > 0 else 0

    return DatasetStats(
        total_count=total_count,
        avg_char_length=avg_length,
        math_count=math_count,
        generic_count=generic_count,
        malformed_count=malformed_count,
    )
