# Legacy Warnings / Known Limitations

This document is a blunt list of “gotchas” and incomplete parts in the current repository state, so readers do not assume features exist when they don’t.

## Incomplete pipeline stages

- `training/cli.py`:
  - `run(...)` is a placeholder (no training loop yet).
  - `merge(...)` is a placeholder (no merge implementation yet).
- `inference/cli.py`:
  - `run(...)` is a placeholder (no inference/metrics yet).

## Dataset generator language and content

- `dataset/generator.py` currently generates **Russian** instructions and questions.
- “Generic” samples are a small fixed list; the dataset is intentionally simplistic.

## Validation heuristics

- `dataset/writer.validate(...)` uses:
  - a simplified regex to extract the embedded JSON from the `### Response:` section
  - a heuristic math-vs-generic classifier based on presence of arithmetic operators in the text

These are fine for the current synthetic format but may need to be upgraded for:
- longer JSON objects,
- multi-line JSON,
- more diverse prompt formats.

## Training configuration assumptions

- LoRA config in `training/model_loader.py` is currently hard-coded.
- Tokenization uses `padding="max_length"` and sets labels equal to `input_ids`.
- Device selection defaults to `cuda`/`mps`/`cpu`, but **training stability and performance** depend heavily on hardware and model size.

## Documentation caveat

Docs under `docs/` describe the intended architecture as well as the parts that are currently implemented; incomplete modules are explicitly called out in the relevant sections.

