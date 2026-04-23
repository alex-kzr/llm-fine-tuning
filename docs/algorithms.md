# Algorithms

This document describes the key algorithms implemented (and planned) in the project.

## 1) Synthetic dataset generation

Location: `dataset/generator.py`

### Goal
Generate instruction-following examples where the target output is a strict JSON object.

### Procedure (per sample)

1. Randomly choose sample type:
   - ~60% math
   - ~40% “generic knowledge”
2. Create:
   - `instruction`: one of several templates asking to respond in JSON
   - `input`: a question string
   - `answer`: the plain-text answer
3. Build the target JSON object:
   - `{"model":"tinyllama","type":"response","message": "<answer>"}`
4. Format a single training text block:
   - `### Instruction:\n...\n\n### Input:\n...\n\n### Response:\n{...json...}\n`
5. Return `{"text": "<block>"}`.

### Notes
- JSON is emitted with `ensure_ascii=False` to preserve non-ASCII characters.
- Random seed can be provided for reproducibility.

## 2) JSONL writing

Location: `dataset/writer.py` (`save(...)`)

### Goal
Persist generated samples to a JSONL file and ensure the output directory exists.

### Procedure
- Create parent directories as needed.
- For each sample:
  - Validate it has a `"text"` key.
  - Write one JSON object per line.

## 3) Dataset validation (embedded JSON schema check)

Location: `dataset/writer.py` (`validate(...)`)

### Goal
Check that:
- each JSONL line parses,
- each line contains `"text"`,
- the `### Response:` section contains a parseable JSON object,
- the embedded JSON object contains required keys: `model`, `type`, `message`.

### Procedure (per non-empty line)

1. Parse the outer JSON object.
2. Extract `text` and count its length.
3. Find the embedded JSON in `### Response:` via regex.
4. Parse the embedded JSON and check required keys.
5. Classify sample as “math” vs “generic” using a heuristic:
   - if `+`, `*`, `-`, `/` appears in `text` → math
   - otherwise → generic

### Output
Returns `DatasetStats` with counts and average text length.

### Limitations
- Regex extraction is intentionally simple and may fail for more complex/long JSON objects.
- The math/generic classifier is heuristic-based and can misclassify.

## 4) Dataset tokenization for training

Location: `training/data_loader.py` (`load_and_tokenize(...)`)

### Goal
Convert JSONL samples into token IDs suitable for causal language model training.

### Procedure

- Load JSONL via `datasets.load_dataset("json", data_files=path)`.
- Tokenize each example’s `"text"` with:
  - truncation enabled
  - `padding="max_length"`
  - `max_length` from config (default 512)
- Set `labels = input_ids.copy()` for causal LM loss.
- Remove the original `"text"` column.

## 5) LoRA application

Location: `training/model_loader.py`

### Steps

1. Load tokenizer (`AutoTokenizer`) and set:
   - `pad_token = eos_token`
2. Load base model (`AutoModelForCausalLM`):
   - attempt `device_map="auto"`
   - fall back to explicit `.to(device)` if needed (common on MPS)
3. Apply LoRA via `peft.get_peft_model(...)` with a fixed `LoraConfig`.

## 6) Device selection

Location: `utils/device.py`

### Procedure

1. If `DEVICE` is set in environment → use it.
2. Else:
   - if `torch.cuda.is_available()` → `cuda`
   - else if MPS is available and built → `mps`
   - else → `cpu`

## Planned algorithms (not implemented yet)

Location: `training/cli.py`, `inference/cli.py`

- Training loop with logging, checkpointing, and adapter saving.
- Inference routine that:
  - formats prompts consistently,
  - generates model output,
  - extracts the JSON object robustly,
  - calculates validity metrics over a stress-test question set.

