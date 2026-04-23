# IQA-02 - Create comprehensive documentation and run instructions

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Author the project README so a new user can set up the environment, run all four CLI commands, interpret metrics, and troubleshoot common failures across Mac/CUDA/CPU.

## Context
Deliverables include "Instructions to run locally". Without this, the pipeline is hard to reproduce and the value of the implementation is diminished.

## Requirements
- `README.md` at repo root covering:
  - **Overview**: what the project does (LoRA fine-tuning for strict JSON output)
  - **Architecture**: module diagram (`dataset/`, `training/`, `inference/`, `utils/`)
  - **Setup**: Python version (3.10+ recommended), `pip install -r requirements.txt`, first-run TinyLlama download note
  - **Configuration**: `.env` / CLI overrides table (one row per config key)
  - **Device-specific notes**:
    - Mac (MPS): batch_size=1, fp16 disabled
    - NVIDIA (CUDA): recommended batch_size, fp16/bf16 options
    - CPU: expect long training time
  - **CLI usage**: each of `generate-dataset`, `train`, `test`, `merge` with example invocations
  - **Expected metrics**: loss trajectory ~1.5 → ~0.7, ≥80% valid JSON in stress test
  - **Troubleshooting**: common errors (OOM, tokenizer pad_token, MPS op not implemented, PEFT version mismatch)
  - **References**: pointer to LoRA paper and TinyLlama model card

## Implementation Notes
- Include a short "Quickstart" at the top: 5 commands to go from clone to merged model
- Link to `board/plan.md` for task breakdown
- Keep commands copy-pasteable (no line continuations that break)

## Testing
- [ ] Walk through README on a clean machine (or fresh venv)
- [ ] Every command in README executes
- [ ] Troubleshooting entries match real errors encountered during IQA-01

## Definition of Done
- [ ] README.md committed
- [ ] Quickstart validated on at least one device
- [ ] Troubleshooting section reflects actual pain points
- [ ] Links to board artifacts work

## Affected Files / Components
- README.md
- (references) board/plan.md, board/tasks.md

## Risks / Dependencies
- IQA-01 output feeds troubleshooting section
- Project config decisions (PSI-01) must be stable before documenting

## Validation Steps
1. Fresh clone → follow README quickstart → smoke pipeline green
2. Review by a teammate (or self-review after a day)
3. Confirm no broken links or stale command references

## Follow-ups
- Add a CHANGELOG if project gains further iterations
