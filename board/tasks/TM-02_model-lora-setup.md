# TM-02 - Build LoRA configuration and model setup

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Load the TinyLlama base model and attach a LoRA adapter with the exact hyperparameters specified so the training loop updates only adapter weights.

## Context
LoRA allows efficient fine-tuning by training small rank-decomposition matrices attached to specific attention projections. Spec: `r=8`, `lora_alpha=16`, `target_modules=["q_proj","v_proj"]`, `lora_dropout=0.05`, `bias="none"`, `task_type="CAUSAL_LM"`.

## Requirements
- Create `training/model_loader.py` with:
  - `load_tokenizer(model_name) -> AutoTokenizer` with `pad_token = eos_token`
  - `load_base_model(model_name) -> AutoModelForCausalLM` with `device_map="auto"` and `use_cache=False`
  - `apply_lora(model) -> PeftModel` using `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")` + `get_peft_model`
  - `build_training_model(cfg) -> (tokenizer, peft_model)` convenience wrapper
- Log trainable parameter count (`model.print_trainable_parameters()`)

## Implementation Notes
- `device_map="auto"` may not behave on MPS — allow fallback to `.to("mps")`
- For MPS, load in fp32 (fp16 causes issues); for CUDA, fp16/bf16 per device defaults
- Respect config for model name — do NOT hardcode

## Testing
- [ ] Unit with a tiny stub model: LoRA config applied, trainable params > 0 but < base
- [ ] Integration: loads TinyLlama on target device (slow test, optional)
- [ ] Verify `use_cache` is False

## Definition of Done
- [ ] `training/model_loader.py` implemented
- [ ] LoRA config matches spec exactly
- [ ] Trainable-param report shows LoRA-only weights
- [ ] Works on MPS, CUDA, CPU

## Affected Files / Components
- training/model_loader.py
- Consumed by TM-03 and IV-01 (for adapter loading)

## Risks / Dependencies
- PEFT API drift across versions — pin version in `requirements.txt`
- TinyLlama weights download requires internet on first run

## Validation Steps
1. Load tokenizer, confirm `pad_token == eos_token`
2. Load base model, confirm `config.use_cache == False`
3. Apply LoRA, confirm trainable params are a small fraction of total

## Follow-ups
- TM-03 feeds this into Trainer
