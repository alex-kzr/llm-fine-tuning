# PSI-04 - Add hardware detection and device management

## Status
- [ ] To Do
- [ ] In Progress
- [x] Done

## Purpose
Auto-detect the best available compute device (CUDA → MPS → CPU) and centralize device-specific defaults so training and inference adapt to Mac, NVIDIA GPU, or CPU without code changes.

## Context
Spec requires hardware support across Mac (MPS), NVIDIA (CUDA), and CPU fallback. Batch size differs per device (Mac: 1, GPU: configurable).

## Requirements
- Create `utils/device.py` with:
  - `detect_device() -> str` returning one of `"cuda"`, `"mps"`, `"cpu"`
  - `device_defaults(device) -> dict` returning `{"batch_size": ..., "fp16": ...}`
    - Mac (mps): batch_size=1, fp16=False
    - CUDA: batch_size=configurable (default 4), fp16 may be True
    - CPU: batch_size=1, fp16=False
  - `to_device(model, device)` helper (uses `torch` device map or `.to()`)
- Config (`utils/config.py`) reads device via `detect_device()` if not overridden
- Log selected device + reason at startup

## Implementation Notes
- CUDA check: `torch.cuda.is_available()`
- MPS check: `torch.backends.mps.is_available() and torch.backends.mps.is_built()`
- Respect explicit `DEVICE` env var override
- For `device_map="auto"` in HF, this module still records the canonical choice for logging

## Testing
- [ ] Unit test with monkeypatched torch to simulate each device
- [ ] Unit test `device_defaults` values
- [ ] Manual: run on target machine and verify correct device is selected

## Definition of Done
- [ ] `utils/device.py` returns correct device on current machine
- [ ] Defaults propagate into config
- [ ] Device logged at CLI startup

## Affected Files / Components
- utils/device.py
- utils/config.py (consumes device detection)
- training/trainer.py and inference modules read effective device

## Risks / Dependencies
- MPS support in PyTorch has bugs for some ops — document known fallbacks
- `device_map="auto"` in HF may not cooperate with MPS

## Validation Steps
1. On Mac: detect returns `"mps"`, batch_size default = 1
2. On CUDA box: returns `"cuda"`
3. Force `DEVICE=cpu`: returns `"cpu"` regardless of hardware

## Follow-ups
- Training (TM-03) reads batch size from device defaults
