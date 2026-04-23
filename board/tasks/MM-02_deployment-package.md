# MM-02 - Create deployment package

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Turn the merged model at `./tinyllama-merged` into a self-contained deployable bundle that downstream consumers can use without the PEFT dependency or the training pipeline code.

## Context
After MM-01 merges LoRA into base weights, the resulting model is a standard HF causal-LM. Packaging it cleanly makes integration into production apps straightforward.

## Requirements
- Produce `./tinyllama-merged/README.md` describing:
  - Model lineage (base + LoRA task)
  - Expected JSON output schema
  - Hardware requirements
  - Usage example
- Produce `./tinyllama-merged/inference_example.py`:
  - Loads the merged model + tokenizer (no PEFT import)
  - Formats a prompt in the training template
  - Generates and prints output
- Optional: produce a `model_card.md` (HF-style) with training config, metrics, and known limitations
- Verify the merged model loads and generates valid JSON without the adapter directory

## Implementation Notes
- Keep `inference_example.py` minimal — copy-pasteable into a new project
- Do not require `peft` at runtime
- Document tokenizer + generation params (`max_new_tokens=150`, `temperature=0.7`, `top_p=0.9`)

## Testing
- [ ] Verify `inference_example.py` runs standalone from `./tinyllama-merged/`
- [ ] Verify a fresh venv without `peft` installed can still load and infer
- [ ] Verify output is valid JSON for a sample prompt

## Definition of Done
- [ ] README.md present in merged directory
- [ ] inference_example.py present and runnable
- [ ] Confirmed no PEFT dependency needed
- [ ] Sample inference produces valid JSON

## Affected Files / Components
- ./tinyllama-merged/README.md
- ./tinyllama-merged/inference_example.py
- (optional) ./tinyllama-merged/model_card.md

## Risks / Dependencies
- MM-01 must complete successfully
- Tokenizer files must be saved alongside model (MM-01 handles)

## Validation Steps
1. `cd ./tinyllama-merged && python inference_example.py`
2. Create a fresh venv with only `transformers` + `torch`, repeat — should work
3. Extract JSON from output, parse, validate schema

## Follow-ups
- IQA-02 references this package in documentation
