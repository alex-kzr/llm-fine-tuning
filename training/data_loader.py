"""Dataset loading and tokenization for training."""

from __future__ import annotations

import re

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


_RESPONSE_MARKER = re.compile(r"###\s*Response\s*:\s*", re.IGNORECASE)


def load_and_tokenize(
    tokenizer: PreTrainedTokenizer,
    path: str,
    max_length: int = 512
) -> Dataset:
    """Load and tokenize dataset for training.

    Args:
        tokenizer: HuggingFace tokenizer with pad_token set.
        path: Path to JSONL dataset file.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset with input_ids, attention_mask, labels columns.
    """
    # Load JSONL dataset
    dataset = load_dataset("json", data_files=path)["train"]

    def tokenize(batch):
        """Tokenize a batch and mask labels to train only on the response."""
        texts: list[str] = batch["text"]
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        pad_token_id = tokenizer.pad_token_id
        labels: list[list[int]] = []
        for text, input_ids in zip(texts, tokens["input_ids"], strict=True):
            label = list(input_ids)

            match = _RESPONSE_MARKER.search(text)
            if match:
                prefix = text[: match.end()]
                prefix_ids = tokenizer(
                    prefix,
                    truncation=True,
                    padding=False,
                    max_length=max_length,
                )["input_ids"]
                prefix_len = min(len(prefix_ids), max_length)
                for i in range(prefix_len):
                    label[i] = -100

            if pad_token_id is not None:
                for i, tok in enumerate(input_ids):
                    if tok == pad_token_id:
                        label[i] = -100

            labels.append(label)

        tokens["labels"] = labels
        return tokens

    # Tokenize and remove text column
    tokenized = dataset.map(tokenize, remove_columns=["text"], batched=True)

    return tokenized
