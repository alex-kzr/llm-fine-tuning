# 📋 Tasks — LoRA Fine-Tuning System for TinyLlama
---

## Phase 1 — Project Structure & Infrastructure

### PSI-01 Create project structure and base configuration
Scaffold the modular project layout (`dataset/`, `training/`, `inference/`, `utils/`, `main.py`) with `__init__.py` files, `requirements.txt`, and the central `utils/config.py` exposing model name, dataset size, batch size, gradient accumulation, epochs, learning rate, max_length, paths, and device — all overridable via `.env` or CLI flags, no hardcoded values elsewhere.
→ [PSI-01_project-structure.md](./tasks/PSI-01_project-structure.md)

### PSI-02 Implement logging and monitoring infrastructure
Add `utils/logging.py` with a standard configured logger (console + file handler under `./logs/`) used by training, inference, and CLI modules. Log training loss per 10 steps, eval results, device info, and CLI command lifecycle.
→ [PSI-02_logging-infrastructure.md](./tasks/PSI-02_logging-infrastructure.md)

### PSI-03 Create CLI interface framework
Build `main.py` with an argparse subparser skeleton exposing `generate-dataset`, `train`, `test`, and `merge` commands that load config, configure logging, and dispatch to the corresponding modules. Provide `--help` for every subcommand and shared flags (`--config`, `--verbose`).
→ [PSI-03_cli-framework.md](./tasks/PSI-03_cli-framework.md)

### PSI-04 Add hardware detection and device management
Implement `utils/device.py` that auto-detects CUDA → MPS → CPU (respecting config override), reports selected device in logs, applies device-specific defaults (e.g., batch_size=1 on Mac, fp16 off by default), and exposes a single helper consumed by training and inference.
→ [PSI-04_device-management.md](./tasks/PSI-04_device-management.md)

---

## Phase 2 — Dataset Generation

### DG-01 Implement dataset generation module
Create `dataset/generator.py` with `generate_sample()` producing math (`a+b`, `a*b`) and general-knowledge samples wrapped in the `### Instruction / ### Input / ### Response: {JSON}` text format, and `generate_dataset(n)` building N samples. JSON schema: `{"model": MODEL_NAME, "type": "response", "message": answer}`.
→ [DG-01_dataset-generation.md](./tasks/DG-01_dataset-generation.md)

### DG-02 Create dataset saving and validation
Create `dataset/writer.py` that writes the generated samples to `dataset.jsonl` (one JSON object per line, UTF-8, `ensure_ascii=False`), plus a validator that re-reads the file, asserts each row parses and matches the expected schema, and reports dataset statistics (count, avg length, math vs. generic ratio).
→ [DG-02_dataset-writer-validator.md](./tasks/DG-02_dataset-writer-validator.md)

---

## Phase 3 — Training Module

### TM-01 Implement tokenization and dataset loading
Create `training/data_loader.py` that loads `dataset.jsonl` via `datasets.load_dataset("json", ...)`, applies a `tokenize(example)` function (`truncation=True`, `padding="max_length"`, `max_length=512`), sets `tokens["labels"] = tokens["input_ids"].copy()`, and removes the raw `text` column.
→ [TM-01_data-loader.md](./tasks/TM-01_data-loader.md)

### TM-02 Build LoRA configuration and model setup
Create `training/model_loader.py` that loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` + tokenizer (`pad_token = eos_token`), disables `use_cache`, and wraps the model with `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")` via `get_peft_model`.
→ [TM-02_model-lora-setup.md](./tasks/TM-02_model-lora-setup.md)

### TM-03 Create training script with configurable hyperparameters
Assemble `training/trainer.py` integrating TM-01 + TM-02, configure `TrainingArguments` (`output_dir=./tinyllama-json`, `per_device_train_batch_size` from config, `gradient_accumulation_steps=4`, `num_train_epochs=2`, `learning_rate=2e-4`, `logging_steps=10`, `save_strategy="epoch"`, `fp16=False`), run `Trainer.train()`, and save the adapter.
→ [TM-03-training-script.md](./tasks/TM-03-training-script.md)

---

## Phase 4 — Inference & Validation

### IV-01 Implement inference module with JSON extraction
Load base model + LoRA adapter from `./tinyllama-json`, format prompts with the Instruction/Input/Response template, generate with `max_new_tokens=150`, `temperature=0.7`, `top_p=0.9`, and extract the first JSON object from output via regex + `json.loads` with safe failure.
→ [IV-01-inference-json-extraction.md](./tasks/IV-01-inference-json-extraction.md)

### IV-02 Create stress testing suite
Run the loaded model against 10+ diverse Russian-language questions (internet, gravity, AI, CPU, etc.), print per-question raw output and per-question validity verdict, and aggregate a valid-JSON counter.
→ [IV-02-stress-testing.md](./tasks/IV-02-stress-testing.md)

### IV-03 Build validation and metrics reporting
Compute `valid_json / total * 100%`, enforce the ≥80% success threshold, and emit a final summary to console and log file including per-question breakdown and final pass/fail verdict.
→ [IV-03-validation-metrics.md](./tasks/IV-03-validation-metrics.md)

---

## Phase 5 — Merging & Deployment

### MM-01 Implement LoRA adapter merging
Load base model + adapter from `./tinyllama-json`, run `merge_and_unload()`, and save the merged standalone model + tokenizer to `./tinyllama-merged`.
→ [MM-01-lora-merging.md](./tasks/MM-01-lora-merging.md)

### MM-02 Create deployment package
Produce a minimal deployable bundle from `./tinyllama-merged`: a standalone inference example script (no PEFT dependency), a short README describing usage and hardware requirements, and verification that the merged model loads and produces valid JSON without the adapter.
→ [MM-02_deployment-package.md](./tasks/MM-02_deployment-package.md)

---

## Phase 6 — Integration & QA

### IQA-01 Perform end-to-end integration testing
Run the full `generate-dataset → train → test → merge` pipeline on a small dataset (e.g., 50 samples, 1 epoch) against all supported devices available locally, verify CLI wiring, exit codes, and that outputs of each stage feed the next without manual intervention.
→ [IQA-01_e2e-integration-test.md](./tasks/IQA-01_e2e-integration-test.md)

### IQA-02 Create comprehensive documentation and run instructions
Author `README.md` with project overview, architecture diagram, setup (Python version, `requirements.txt` install, device-specific notes for Mac/CUDA/CPU), CLI usage, expected metrics (loss trajectory, ≥80% JSON accuracy), and troubleshooting.
→ [IQA-02_documentation.md](./tasks/IQA-02_documentation.md)

---
