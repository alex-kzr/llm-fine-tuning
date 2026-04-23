# PSI-02 - Implement logging and monitoring infrastructure

## Status
- [ ] To Do
- [ ] In Progress
- [x] Done

## Purpose
Provide consistent structured logging across training, inference, and CLI so progress, loss values, and evaluation results are captured and reviewable after runs.

## Context
Non-functional requirements demand logs for training progress, loss values, and evaluation results. A shared logger avoids ad-hoc `print()` calls and enables persistence of training runs.

## Requirements
- Create `utils/logging.py` with `get_logger(name)` helper
- Console handler (INFO by default; DEBUG under `--verbose`)
- File handler writing to `./logs/{YYYY-MM-DD_HH-MM-SS}_{command}.log`
- Formatter: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`
- Integrate with HuggingFace Trainer via callback or by attaching to `transformers` logger
- Log: device info, config snapshot at start, per-10-step loss, eval results, CLI command lifecycle

## Implementation Notes
- Stdlib `logging` is sufficient; avoid third-party deps
- Lazy-create `logs/` directory if absent
- Root logger should not be configured globally by imports — only when CLI runs

## Testing
- [ ] Unit test: logger creation returns configured instance
- [ ] Unit test: file created under `logs/` after emitting a record
- [ ] Manual: verbose flag toggles DEBUG level

## Definition of Done
- [ ] `utils/logging.py` implemented
- [ ] All modules use `get_logger(__name__)`
- [ ] Log files appear under `./logs/` on each run
- [ ] No stray `print()` in production paths (CLI UX prints allowed)

## Affected Files / Components
- utils/logging.py
- CLI (PSI-03) wires logging at entry
- training/trainer.py logs loss
- inference modules log results

## Risks / Dependencies
- Duplicate handlers on repeated imports — guard with `logger.handlers` check

## Validation Steps
1. Trigger a CLI command, confirm log file created with expected name
2. Trigger training with 1 epoch on tiny dataset, confirm loss lines appear every 10 steps
3. Confirm `--verbose` increases verbosity

## Follow-ups
- Plug into TM-03 trainer callback
