# DG-02 - Create dataset saving and validation

## Status
- [ ] To Do
- [ ] In Progress
- [x] Done

## Purpose
Persist generated samples to `dataset.jsonl` in a format `datasets.load_dataset` can read, and validate structural correctness before training to catch malformed samples early.

## Context
Training reads `dataset.jsonl` via `load_dataset("json", data_files=...)`. A malformed file will either fail loudly or (worse) silently degrade training. This task guards the handoff.

## Requirements
- Create `dataset/writer.py` with:
  - `save(dataset, path)` writing one JSON object per line (UTF-8, `ensure_ascii=False`)
  - `validate(path) -> DatasetStats` that:
    - Re-reads each line, parses JSON
    - Asserts each object has `"text"` key (str, non-empty)
    - Extracts the embedded JSON from `### Response:` and validates schema `{model, type, message}`
    - Returns stats: total count, avg char length, math/generic counts, malformed count
  - Raises `ValueError` with clear message on validation failure
- Integrated into CLI: `generate-dataset` runs generate → save → validate and logs stats

## Implementation Notes
- Use `json.loads(line)` per row, not `json.load(file)`
- Regex for extracting embedded JSON: `re.search(r"\{[^{}]*\"model\".*?\}", text, re.DOTALL)` — or parse from `### Response:\n` onward
- Log stats via PSI-02 logger

## Testing
- [ ] Unit: save then validate round-trips cleanly
- [ ] Unit: intentionally malformed row → validation fails with informative error
- [ ] Unit: stats counts are accurate

## Definition of Done
- [ ] `dataset.jsonl` written correctly for N samples
- [ ] Validator catches structural errors
- [ ] Stats logged via shared logger
- [ ] CLI `generate-dataset` wires generate → save → validate

## Affected Files / Components
- dataset/writer.py
- dataset/cli.py (if separate) wires into main.py
- Consumes DG-01 generator

## Risks / Dependencies
- UTF-8 encoding bugs on Windows — open files with explicit `encoding="utf-8"`

## Validation Steps
1. `python main.py generate-dataset --size 50` → file exists, 50 lines
2. Validator reports 0 malformed, correct stats
3. Corrupt one line manually → validator fails loudly

## Follow-ups
- TM-01 loads this file
