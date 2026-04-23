# PSI-01 - Create project structure and base configuration

## Status
- [ ] To Do
- [ ] In Progress
- [x] Done

## Purpose
Establish the modular project skeleton and a single source of truth for configuration so all downstream modules (dataset, training, inference, merging) share the same parameters without hardcoded values.

## Context
The system is modular (dataset/, training/, inference/, utils/) and must run on Mac/GPU/CPU with configurable hyperparameters (model name, batch size, epochs, dataset size, device). This task provides the directory layout and `utils/config.py` that everything else consumes.

## Requirements
- Create directories: `dataset/`, `training/`, `inference/`, `utils/`, `logs/`, each with `__init__.py`
- Create `main.py` placeholder at project root
- Create `requirements.txt` with: transformers, peft, datasets, torch, accelerate, python-dotenv
- Create `utils/config.py` exposing a dataclass (or pydantic-settings) `Config` with:
  - `model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`
  - `dataset_size: int = 300`
  - `dataset_path: str = "dataset.jsonl"`
  - `adapter_output: str = "./tinyllama-json"`
  - `merged_output: str = "./tinyllama-merged"`
  - `batch_size: int` (device-aware default)
  - `gradient_accumulation_steps: int = 4`
  - `epochs: int = 2`
  - `learning_rate: float = 2e-4`
  - `max_length: int = 512`
  - `device: str` (auto via PSI-04)
- Support `.env` overrides and CLI overrides
- Provide `load_config()` helper

## Implementation Notes
- Prefer `dataclasses` + `python-dotenv` for simplicity; pydantic-settings acceptable
- No hardcoded hyperparameters outside config
- Expose a single import: `from utils.config import load_config`

## Testing
- [ ] Unit tests for default values
- [ ] Unit tests for `.env` overrides
- [ ] Manual: import config from every module

## Definition of Done
- [ ] Directory layout exists and is importable
- [ ] `utils/config.py` returns a typed config object
- [ ] `.env.example` committed
- [ ] `requirements.txt` complete
- [ ] No hardcoded values in downstream modules

## Affected Files / Components
- utils/config.py
- requirements.txt
- .env.example
- main.py (scaffold)
- dataset/__init__.py, training/__init__.py, inference/__init__.py, utils/__init__.py

## Risks / Dependencies
- Wrong defaults propagate everywhere — double-check values against the prompt
- PEFT/transformers version drift can break LoRA API

## Validation Steps
1. Run `python -c "from utils.config import load_config; print(load_config())"`
2. Override a value via `.env`, re-run, confirm override takes effect
3. Verify all four module dirs are importable

## Follow-ups
- PSI-04 fills in device auto-detection
- PSI-02 adds logging alongside config
