# Project Overview

This repository is a small, modular Python project for fine-tuning a pre-trained LLM with **LoRA** so that, at inference time, the model reliably returns **strict JSON** responses of the form:

```json
{"model":"tinyllama","type":"response","message":"..."}
```

The project is designed to run locally on:
- **Mac (MPS)** where supported by PyTorch
- **NVIDIA GPU (CUDA)**
- **CPU** as a fallback

## What it does

The intended end-to-end workflow is:

1. **Generate** a synthetic instruction-following dataset (`generate-dataset`)
2. **Train** a LoRA adapter on a base model (`train`)
3. **Evaluate** / run inference and JSON-validity checks (`test`)
4. **Merge** the adapter into the base model weights for deployment (`merge`)

## Current status (repo reality)

- Dataset generation and validation are implemented under `dataset/`.
- Tokenization and LoRA model construction utilities exist under `training/`.
- `train`, `test`, and `merge` CLI commands are wired but currently contain placeholders in their CLI adapters.

See: `docs/legacy-warnings.md` for limitations and TODO notes.

## Repository layout

- `main.py` — CLI entrypoint and subcommand router
- `utils/` — configuration (`.env`), device detection, logging
- `dataset/` — synthetic dataset generation, JSONL writing, validation
- `training/` — dataset loading/tokenization; base model + LoRA adapter setup
- `inference/` — (planned) inference + JSON extraction/metrics
- `docs/` — project documentation
- `board/` — project plan and task descriptions

## Outputs (files on disk)

The pipeline writes artifacts to these default paths (override via `.env` or CLI where supported):

- Dataset: `dataset.jsonl`
- LoRA adapter: `./tinyllama-json`
- Merged model: `./tinyllama-merged`

## Next reading

- Architecture: `docs/architecture.md`
- Domain concepts: `docs/domain-model.md`
- How it works: `docs/algorithms.md`
- How to run: `docs/usage-guide.md`

