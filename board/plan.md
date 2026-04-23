# Feature: LoRA Fine-Tuning System for TinyLlama (Strict JSON Output)

Build a complete, modular application that fine-tunes a small pre-trained LLM (TinyLlama 1.1B-Chat) with LoRA so the model always returns strict JSON responses in the form `{"model": "tinyllama", "type": "response", "message": "..."}`. The system must run locally on Mac (MPS), NVIDIA GPU (CUDA), or CPU, expose a CLI with `generate-dataset`, `train`, `test`, and `merge` commands, and include synthetic dataset generation, LoRA training, inference with JSON validation, and adapter merging — all driven by a central configuration module and supported by logging.

## Phase 1: Project Structure & Infrastructure (PSI-01 to PSI-04)
Lay down the modular project skeleton (`dataset/`, `training/`, `inference/`, `utils/`), the central configuration system (`utils/config.py` covering model name, batch size, epochs, dataset size, device selection, I/O paths), structured logging for training/evaluation, the CLI framework (argparse-based `main.py`), and the hardware detection layer that auto-selects MPS/CUDA/CPU and exposes a single `device` abstraction to the rest of the pipeline.

## Phase 2: Dataset Generation (DG-01 to DG-02)
Implement the synthetic dataset generator that emits math and general-knowledge samples in the `### Instruction / ### Input / ### Response: {JSON}` format (schema: `{"model": "tinyllama", "type": "response", "message": ...}`), and the JSONL writer/validator that persists a configurable number of rows (default 300–1000) to `dataset.jsonl` and validates structural correctness before training.

## Phase 3: Training Module (TM-01 to TM-03)
Build the training pipeline: JSONL tokenization and dataset loading (`padding="max_length"`, `truncation=True`, `max_length=512`, `labels = input_ids.copy()`), base model + LoRA adapter setup (`r=8`, `alpha=16`, `target_modules=["q_proj","v_proj"]`, `dropout=0.05`, `bias="none"`, `task_type="CAUSAL_LM"`), and the trainer script that runs 2 epochs at LR 2e-4, logs loss every 10 steps, saves per epoch, and writes the adapter to `./tinyllama-json`.

## Phase 4: Inference & Validation (IV-01 to IV-03)
Load base model + LoRA adapter for inference, generate responses for formatted prompts, extract JSON from raw output via regex + `json.loads`, run a stress-test over 10+ diverse questions, and compute accuracy metrics. Success: ≥80% valid JSON across the stress test.

## Phase 5: Merging & Deployment (MM-01 to MM-02)
Merge the trained LoRA adapter into the base weights via `merge_and_unload()`, save the standalone merged model + tokenizer to `./tinyllama-merged`, and produce a deployable package (merged artifact + README + minimal inference script) suitable for distribution without the PEFT dependency.

## Phase 6: Integration & QA (IQA-01 to IQA-02)
Run end-to-end integration testing across the full `generate-dataset → train → test → merge` flow on a small dataset, verify CLI wiring and error paths, and author the run-instructions documentation (environment setup, device-specific notes for Mac/CUDA/CPU, troubleshooting, expected metrics).
