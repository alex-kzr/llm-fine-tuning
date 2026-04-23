# Usage Guide

This guide describes how to run the CLI locally and what each command is expected to do.

## Quickstart

### Setup configuration

Configuration is loaded from `.env` (default) and environment variables. You can also point to an additional env file via `--env-file` (alias: `--config`).

Supported keys (see `utils/config.py`):

| Key | Meaning | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face Hub token (optional) | unset |
| `MODEL_NAME` | HF base model identifier | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `DATASET_SIZE` | number of synthetic samples | `300` |
| `DATASET_PATH` | JSONL dataset path | `dataset.jsonl` |
| `QUESTIONS_PATH` | questions/prompts file for `test` | `test_dataset.jsonl` |
| `ADAPTER_OUTPUT` | adapter output dir | `./tinyllama-json` |
| `MERGED_OUTPUT` | merged model output dir | `./tinyllama-merged` |
| `BATCH_SIZE` | training batch size | `8` |
| `GRADIENT_ACCUMULATION_STEPS` | grad accum steps | `4` |
| `EPOCHS` | epochs | `2` |
| `LEARNING_RATE` | learning rate | `2e-4` |
| `MAX_LENGTH` | max tokens | `512` |
| `DEVICE` | `cuda` / `mps` / `cpu` | auto-detected |

Example `.env`:

```env
HF_TOKEN=hf_...
DEVICE=cuda
DATASET_SIZE=500
BATCH_SIZE=4
```


### Windows PowerShell

Run everything from the project root.

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Generate a dataset (JSONL):

```powershell
python .\main.py generate-dataset --size 300
```

By default it writes to `.\dataset.jsonl`.

Run LoRA training (saves the adapter to `ADAPTER_OUTPUT`, default: `.\tinyllama-json`):

```powershell
python .\main.py train --dataset dataset.jsonl --epochs 2 --batch-size 4
```

Test “strict JSON only” on questions (default: `test_dataset.jsonl`):

```powershell
python .\main.py test --adapter .\tinyllama-json --questions-file test_dataset.jsonl
```

This command returns exit code `0` **only** if every response is valid JSON with no extra text.

(Optional) Merge the adapter into the base model:

```powershell
python .\main.py merge --adapter .\tinyllama-json --output .\tinyllama-merged
```

Note: the first run of `train` / `test` / `merge` downloads the base model from Hugging Face (`MODEL_NAME`), which may take time and disk space.

### macOS / Linux

Run everything from the project root.

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate a dataset (JSONL):

```bash
python3 ./main.py generate-dataset --size 300
```

By default it writes to `./dataset.jsonl`.

Run LoRA training (saves the adapter to `ADAPTER_OUTPUT`, default: `./tinyllama-json`):

```bash
python3 ./main.py train --dataset dataset.jsonl --epochs 2 --batch-size 4
```

Test “strict JSON only” on questions (default: `test_dataset.jsonl`):

```bash
python3 ./main.py test --adapter ./tinyllama-json --questions-file test_dataset.jsonl
```

This command returns exit code `0` **only** if every response is valid JSON with no extra text.

(Optional) Merge the adapter into the base model:

```bash
python3 ./main.py merge --adapter ./tinyllama-json --output ./tinyllama-merged
```

Note: the first run of `train` / `test` / `merge` downloads the base model from Hugging Face (`MODEL_NAME`), which may take time and disk space.


## CUDA (GPU) note

`pip install -r requirements.txt` will typically install a **CPU-only** PyTorch build from PyPI. In that case the CLI will log `Detected device: cpu` even if you have an NVIDIA GPU.

To use CUDA, install a **CUDA-enabled** PyTorch wheel first (pick the exact command from the official PyTorch install selector), then install the rest of the requirements:

```bash
pip uninstall -y torch torchvision torchaudio
# Install CUDA-enabled wheels (example shows CUDA 12.1; use the command from pytorch.org for your setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If CUDA wheels are not available for your Python version, create the venv with a supported Python (commonly Python 3.12 on Windows) and retry.



## CLI overview

Run help:

```bash
python main.py --help
python main.py generate-dataset --help
```

### `generate-dataset`

Generates a synthetic JSONL dataset and validates it.

```bash
python main.py generate-dataset --size 300
```

Expected output artifact:
- `dataset.jsonl` (or `DATASET_PATH`)

### `train`

Train a LoRA adapter on the dataset and save it to `ADAPTER_OUTPUT` (plus a tokenizer copy).

```bash
python main.py train --epochs 2 --batch-size 4 --dataset dataset.jsonl
```

### `test`

Run inference with the LoRA adapter and validate that the model outputs **strict JSON only**.

```bash
python main.py test --adapter ./tinyllama-json
```

### `merge`

Merge a LoRA adapter into the base model and save a standalone model directory.

```bash
python main.py merge --adapter ./tinyllama-json --output ./tinyllama-merged
```

## Device notes (quick)

- **CUDA**: usually best performance; enable with `DEVICE=cuda`.
- **MPS**: may require smaller batch sizes; use `DEVICE=mps`.
- **CPU**: works but can be extremely slow for training.

## Troubleshooting and limitations

See:
- `docs/legacy-warnings.md` for known limitations and TODOs.
- `docs/troubleshooting.md` for common setup issues (CUDA, installs, etc.).
