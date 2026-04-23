# TM-01 - Implement tokenization and dataset loading

## Status
- [ ] To Do
- [ ] In Progress
- [x] Done

## Purpose
Convert `dataset.jsonl` into a tokenized `datasets.Dataset` that the HuggingFace `Trainer` can consume for causal-LM fine-tuning with correct `labels`.

## Context
Training expects fixed-length (`max_length=512`) padded sequences with `input_ids`, `attention_mask`, and `labels` (copy of `input_ids`). The `text` column must be removed after tokenization.

## Requirements
- Create `training/data_loader.py` with:
  - `load_and_tokenize(tokenizer, path, max_length=512) -> Dataset`
  - Uses `datasets.load_dataset("json", data_files=path)["train"]`
  - Defines inner `tokenize(example)`:
    - `tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)`
    - Sets `tokens["labels"] = tokens["input_ids"].copy()`
  - Calls `dataset.map(tokenize, remove_columns=["text"])`
  - Returns the tokenized dataset

## Implementation Notes
- Tokenizer must have `pad_token` set (typically `eos_token`) — responsibility of TM-02 but document the contract
- Consider `batched=True` for faster mapping on larger datasets
- Preserve determinism — no shuffling here; Trainer handles that

## Testing
- [ ] Unit: with a small fixture dataset, returned dataset has columns `input_ids`, `attention_mask`, `labels` and no `text`
- [ ] Unit: all sequences have length `max_length`
- [ ] Unit: `labels[i] == input_ids[i]` element-wise

## Definition of Done
- [ ] `training/data_loader.py` implemented
- [ ] Tokenization matches spec
- [ ] `text` column removed, `labels` present
- [ ] Works on 300-sample dataset without OOM

## Affected Files / Components
- training/data_loader.py
- Consumed by TM-03 (trainer)

## Risks / Dependencies
- Missing `pad_token` on tokenizer → padding fails
- `max_length` too high on Mac/CPU → slow training

## Validation Steps
1. Run on fixture of 10 samples, inspect first example shapes
2. Confirm label-input equality
3. Ensure `len(ds)` matches line count

## Follow-ups
- TM-02 provides configured tokenizer
- TM-03 wires this into Trainer
