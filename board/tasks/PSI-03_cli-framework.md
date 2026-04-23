# PSI-03 - Create CLI interface framework

## Status
- [ ] To Do
- [ ] In Progress
- [x] Done

## Purpose
Provide a single entrypoint `main.py` that exposes the four project commands (`generate-dataset`, `train`, `test`, `merge`) and dispatches to the corresponding modules using the central config and shared logger.

## Context
Spec requires CLI with commands: `generate-dataset`, `train`, `test`, `merge`. Example: `python main.py train`.

## Requirements
- Use `argparse` with subparsers for each command
- Shared flags: `--config PATH` (optional override), `--verbose`
- Command `generate-dataset`: optional `--size N`
- Command `train`: optional `--epochs`, `--batch-size`, `--dataset`
- Command `test`: optional `--adapter PATH`, `--questions-file`
- Command `merge`: optional `--adapter PATH`, `--output PATH`
- Each command loads config (PSI-01), initializes logger (PSI-02), resolves device (PSI-04), and calls into its module
- Non-zero exit code on failure; zero on success
- `--help` available at top level and per-command

## Implementation Notes
- Keep `main.py` thin; business logic lives inside module-level functions like `dataset.cli.run(cfg)`
- Module dispatch via subparser `set_defaults(func=...)`

## Testing
- [ ] Unit test: `python main.py --help` exits 0 and shows subcommands
- [ ] Unit test: each subcommand `--help` works
- [ ] Integration: `python main.py generate-dataset --size 10` creates dataset.jsonl

## Definition of Done
- [ ] `main.py` runs and dispatches all four commands
- [ ] Overrides propagate into config
- [ ] Help output is clear and complete
- [ ] Errors are logged with non-zero exit

## Affected Files / Components
- main.py
- dataset/cli.py (thin adapter)
- training/cli.py (thin adapter)
- inference/cli.py (thin adapter)

## Risks / Dependencies
- Argparse subparser wiring is easy to get wrong — unit-test parsing explicitly

## Validation Steps
1. `python main.py --help`
2. `python main.py train --help`
3. `python main.py generate-dataset --size 10` → file created
4. Invalid command exits with code 2 and usage message

## Follow-ups
- Each command module becomes implemented alongside its phase (DG, TM, IV, MM)
