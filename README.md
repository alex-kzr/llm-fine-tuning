# LLM Fine-Tuning (LoRA) for Strict JSON Output

This repository is a small, modular Python CLI project for fine-tuning a pretrained LLM with **LoRA** so that, at inference time, the model reliably returns **strict JSON only** (no extra text), e.g.:

```json
{"model":"tinyllama","type":"response","message":"..."}
```

It is intended to run locally on **CUDA (NVIDIA GPU)**, **MPS (Apple Silicon)**, or **CPU** (slow).

## What’s included

- Synthetic dataset generation and JSONL validation (`generate-dataset`)
- LoRA training for a base chat model (`train`)
- Inference + “strict JSON” validation on a questions file (`test`)
- Adapter merge into base weights for deployment (`merge`)

## Quickstart

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional configuration:

- Copy/edit `.env.example` → `.env`
- Or pass an extra env file via `--env-file`

Run the pipeline:

```bash
python main.py generate-dataset --size 300
python main.py train --dataset dataset.jsonl --epochs 2 --batch-size 4
python main.py test --adapter ./tinyllama-json --questions-file test_dataset.jsonl
python main.py merge --adapter ./tinyllama-json --output ./tinyllama-merged
```

Notes:

- The first run of `train` / `test` / `merge` will download the base model from Hugging Face (see `MODEL_NAME`).
- If you need CUDA, you may need to install a CUDA-enabled PyTorch wheel (see the usage guide).

## Repository layout (high level)

- `main.py` — CLI entrypoint and subcommand router
- `utils/` — config loading (`.env`), device detection, logging
- `dataset/` — synthetic dataset generation, JSONL writing, validation
- `training/` — tokenization + LoRA model build + training + merge
- `inference/` — inference + JSON extraction/validation and metrics
- `docs/` — project documentation
- `board/` — project plan and task descriptions

## Documentation

- Project overview: `docs/project-overview.md`
- Usage / setup: `docs/usage-guide.md`
- Architecture: `docs/architecture.md`
- Algorithms / data format details: `docs/algorithms.md`
- Domain concepts: `docs/domain-model.md`
- Change request template/process: `docs/change-request.md`
- Known limitations / caveats: `docs/legacy-warnings.md`
