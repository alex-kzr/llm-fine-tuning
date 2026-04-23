# MM-01 - Implement LoRA adapter merging

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Implement functionality to merge LoRA adapter weights into the base model, creating a standalone fine-tuned model that doesn't require separate adapter loading. This is essential for production deployment.

## Context
During training, LoRA adapters are small weight additions applied on top of the base model. For production, we merge these weights into a single model file, simplifying deployment and reducing complexity.

## Requirements
- Create `training/merger.py` module
- Implement `merge_adapter(base_model_name, adapter_path, output_path)` function:
  - Load base model (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  - Load LoRA adapter from trained path
  - Call `model.merge_and_unload()`
  - Save merged model to output_path
  - Save tokenizer to output_path
  - Return success status and model info
- Implement validation:
  - Verify adapter_path exists and has valid model
  - Verify base model can be loaded
  - Verify merged model loads successfully
  - Verify merged model has correct architecture
- Implement file management:
  - Create output directory if needed
  - Handle existing files (overwrite or skip)
  - Clean up temporary files if applicable
  - Verify sufficient disk space
- Add logging:
  - Log merge progress
  - Log file sizes (base, adapter, merged)
  - Log reduction in model size compared to full model
  - Log merge duration
- Handle errors gracefully:
  - Catch model loading errors
  - Handle disk space issues
  - Provide helpful error messages

## Implementation Notes
- PEFT's merge_and_unload() combines adapter weights into base model
- Merged model is a standard HuggingFace model
- Output should be approximately same size as base model
- Merging is unidirectional (can't separate after merging)
- Verify merged model by forward pass
- Consider making backup of original base model first
- Merged model should have identical architecture and weights as base

## Testing
- [ ] Test merge completes successfully
- [ ] Test merged model loads with AutoModel
- [ ] Test merged model produces same output as adapter+base
- [ ] Test files are saved correctly
- [ ] Test error handling for missing adapter
- [ ] Test error handling for missing base model
- [ ] Test merged model size is reasonable
- [ ] Test inference works on merged model

## Definition of Done
- [ ] `training/merger.py` created
- [ ] Merge function implemented and working
- [ ] Merged model saves successfully
- [ ] Merged model loads as standalone
- [ ] Merged model produces correct outputs
- [ ] Tokenizer saved with model
- [ ] All files organized properly
- [ ] Logging tracks merge process

## Affected Files / Components
- training/merger.py
- Output: ./tinyllama-merged/ directory with model files
- Uses PEFT and transformers libraries
- Used by MM-02 and CLI

## Risks / Dependencies
- Requires sufficient disk space (2-3GB for model)
- Network required for base model download (if not cached)
- Merge cannot be reversed (kept adapter as backup)
- Merged model performance must match adapter+base

## Validation Steps
1. Train model to completion
2. Merge adapter into base model
3. Load merged model as standalone
4. Generate response with merged model
5. Compare output with adapter+base version
6. Verify file sizes are correct
7. Check model can be loaded by HuggingFace library

## Follow-ups
- MM-02: Create final deployment package
- IQA-01: Test merged model end-to-end
- Optionally: Test merged model quantization for smaller deployment
