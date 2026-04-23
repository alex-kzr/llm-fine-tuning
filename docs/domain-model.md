# Domain Model

This document defines the main “things” (concepts and artifacts) in the project and how they relate.

## Core concepts

### `Config` (runtime configuration)

Defined in `utils/config.py` as a dataclass and populated from `.env` + environment variables.

Key fields:
- `hf_token`: Hugging Face Hub token (optional; enables authenticated downloads)
- `model_name`: HuggingFace model ID (default TinyLlama chat)
- `dataset_size`: number of generated samples
- `dataset_path`: JSONL dataset path
- `adapter_output`: directory to save a LoRA adapter
- `merged_output`: directory to save a merged model
- `batch_size`, `gradient_accumulation_steps`, `epochs`, `learning_rate`, `max_length`
- `device`: `cuda` / `mps` / `cpu`

### Training sample (synthetic instruction dataset)

Produced by `dataset/generator.py` and persisted by `dataset/writer.py`.

In-memory representation:
- A Python dict like `{"text": "<formatted prompt and response>"}`.

On disk (JSONL):
- One JSON object per line with at least the key `"text"`.

### Target JSON response schema

The embedded response in each sample’s `### Response:` block is:

- `model`: string (e.g., `"tinyllama"`)
- `type`: string (e.g., `"response"`)
- `message`: string (the answer content)

### Dataset statistics (`DatasetStats`)

Defined in `dataset/writer.py` and returned by `validate(...)`:

- `total_count`
- `avg_char_length`
- `math_count`
- `generic_count`
- `malformed_count`

### LoRA adapter

An adapter is the set of trainable LoRA weights applied to a base model using `peft`.

In this repo, the LoRA configuration (currently hard-coded) targets:
- `target_modules=["q_proj", "v_proj"]`
- `r=8`, `lora_alpha=16`, `lora_dropout=0.05`

Adapter output is intended to be written to `Config.adapter_output`.

### Merged model

A merged model is the base model with LoRA weights merged into the base weights (no PEFT required at runtime). Intended output directory: `Config.merged_output`.

## Commands (domain actions)

The CLI exposes four domain actions:

- `generate-dataset`: creates and validates `dataset.jsonl`
- `train`: trains a LoRA adapter from dataset (currently placeholder)
- `test`: runs inference and computes JSON validity metrics (currently placeholder)
- `merge`: merges adapter into base weights (currently placeholder)

## Files and directories (artifacts)

- `dataset.jsonl`: synthetic training dataset
- `./tinyllama-json/`: adapter output directory (intended)
- `./tinyllama-merged/`: merged model output directory (intended)
- `.env`: configuration overrides (optional)
