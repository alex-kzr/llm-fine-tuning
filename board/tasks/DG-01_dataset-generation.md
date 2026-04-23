# DG-01 - Implement dataset generation module

## Status
- [ ] To Do
- [ ] In Progress
- [x] Done

## Purpose
Produce synthetic training samples in the exact `### Instruction / ### Input / ### Response: {JSON}` format required for LoRA fine-tuning on strict JSON output behavior.

## Context
Fine-tuning success depends on dataset quality and formatting. The JSON schema is fixed: `{"model": "tinyllama", "type": "response", "message": "..."}`. Samples must mix math and general-knowledge types.

## Requirements
- Create `dataset/generator.py` with:
  - `INSTRUCTIONS` list (e.g., "Ответь строго в JSON формате", "Верни только JSON объект")
  - `MATH_TEMPLATES` list (e.g., "Сколько будет {a}+{b}?", "Чему равно {a} * {b}?")
  - `GENERIC_SAMPLES` list of (question, answer) tuples covering general-knowledge
  - `generate_math_sample()` and `generate_generic_sample()`
  - `generate_sample()` that randomly picks one type
  - `generate_dataset(n: int) -> list[dict]` returning N samples
- Each sample is a dict: `{"text": "### Instruction:\n...\n\n### Input:\n...\n\n### Response:\n{json}\n"}`
- JSON uses `ensure_ascii=False`
- Computed answers for math are correct (addition, multiplication)

## Implementation Notes
- Seed RNG from config for reproducibility (optional `--seed`)
- Keep answers short and deterministic for math
- Aim for roughly balanced math/generic ratio (e.g., 60/40)
- Do NOT use any external API — fully synthetic

## Testing
- [ ] Unit: `generate_dataset(10)` returns 10 dicts each with `"text"` key
- [ ] Unit: parsing the JSON fragment from `### Response:` yields valid object matching schema
- [ ] Unit: math answers are arithmetically correct
- [ ] Unit: at least one generic and one math sample in a batch of 50

## Definition of Done
- [ ] `dataset/generator.py` implemented
- [ ] Samples match exact format required by training
- [ ] Math answers correct
- [ ] Mix of math and generic samples

## Affected Files / Components
- dataset/generator.py
- Consumed by DG-02 (writer/validator) and CLI

## Risks / Dependencies
- Subtle formatting drift (missing newline, extra space) will hurt training — match template exactly

## Validation Steps
1. Generate 100 samples, spot-check 5
2. Extract JSON from the `### Response:` section and validate schema
3. Verify math answers

## Follow-ups
- DG-02 writes and validates the file
