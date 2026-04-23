# IQA-01 - Perform end-to-end integration testing

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Verify the full pipeline (`generate-dataset → train → test → merge`) works end-to-end via the CLI with stage outputs feeding the next stage without manual intervention.

## Context
Every module is unit-tested in isolation. This task validates their integration — CLI wiring, config propagation, file paths, device selection — on a realistic smoke-test run.

## Requirements
- Run on a scaled-down config (50 samples, 1 epoch) to keep runtime manageable
- Sequence:
  1. `python main.py generate-dataset --size 50` → `dataset.jsonl` created, validated
  2. `python main.py train --epochs 1` → adapter saved to `./tinyllama-json`, loss logged
  3. `python main.py test --adapter ./tinyllama-json` → stress test runs, JSON accuracy reported
  4. `python main.py merge` → `./tinyllama-merged` created
- Confirm exit code 0 at every stage
- Repeat on each available local device (MPS, CUDA, CPU) if hardware permits
- Document any stage-specific failures and fix

## Implementation Notes
- Write a shell script `scripts/e2e_smoke.sh` that runs the sequence and fails fast
- Capture logs from each stage under `logs/`
- Consider adding a pytest integration test marker `@pytest.mark.slow` for this

## Testing
- [ ] E2E smoke: all four commands complete successfully
- [ ] Accuracy from step 3 is reported (may be low on 1-epoch run — that's OK here)
- [ ] Merged model directory has expected files

## Definition of Done
- [ ] E2E smoke script exists and is green on at least one target device
- [ ] CLI exit codes correct at every stage
- [ ] Logs captured for each stage
- [ ] Any integration bugs found are fixed

## Affected Files / Components
- scripts/e2e_smoke.sh
- (optional) tests/test_e2e.py

## Risks / Dependencies
- First-run download of TinyLlama requires internet + time
- MPS-specific bugs in ops may surface only here — document workarounds

## Validation Steps
1. Clean working tree (no dataset.jsonl, no tinyllama-json, no tinyllama-merged)
2. Run smoke script
3. Inspect logs and verify artifacts

## Follow-ups
- IQA-02 documents the smoke-run procedure in README
