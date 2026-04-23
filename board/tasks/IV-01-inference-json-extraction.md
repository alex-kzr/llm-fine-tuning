# IV-01 - Implement inference module with JSON extraction

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Create inference capabilities to load the fine-tuned model and generate responses. Implement robust JSON extraction from raw model output using regex and JSON parsing, handling various edge cases.

## Context
After training, the model must generate responses in the specified JSON format. Raw model output includes training prompts and may have formatting variations. Extraction must be robust to extract valid JSON even from imperfect outputs.

## Requirements
- Create `inference/generator.py` module
- Implement `load_model(adapter_path)` function:
  - Load base model (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  - Load LoRA adapter from path
  - Place on correct device
  - Set model.eval()
  - Return ready-for-inference model
- Implement `generate(prompt, **kwargs)` function:
  - Tokenize prompt
  - Call model.generate() with:
    - max_new_tokens: 150
    - temperature: 0.7
    - do_sample: True
    - top_p: 0.9
  - Decode output tokens
  - Return raw text output
- Implement `extract_json(text)` function:
  - Use regex to find JSON object: `\{.*\}`
  - Parse found JSON with json.loads()
  - Handle incomplete JSON gracefully
  - Return dict if valid, None if invalid
  - Log extraction success/failure
- Implement `generate_with_extraction(prompt)` that:
  - Generates response
  - Extracts JSON
  - Returns (raw_output, parsed_json, is_valid)
- Add temperature/sampling parameter control
- Implement batch inference for multiple prompts
- Log inference timing and metrics

## Implementation Notes
- Use regex with DOTALL flag to handle multiline JSON
- Handle cases where JSON is partially complete
- Consider whitespace and formatting variations
- May need to clean output before JSON parsing
- Temperature=0.7 provides good diversity vs consistency
- top_p=0.9 helps avoid nonsensical outputs
- Cache tokenizer to avoid reloading
- GPU inference should be efficient for single samples

## Testing
- [ ] Test model loading and initialization
- [ ] Test inference on sample prompts
- [ ] Test JSON extraction from perfect JSON
- [ ] Test JSON extraction from incomplete JSON
- [ ] Test JSON extraction from invalid JSON
- [ ] Test batch inference
- [ ] Test different temperature settings
- [ ] Test with various prompt formats

## Definition of Done
- [ ] `inference/generator.py` created with all functions
- [ ] Model loads successfully with LoRA adapter
- [ ] Inference generates coherent responses
- [ ] JSON extraction works for valid JSON
- [ ] JSON extraction returns None for invalid JSON
- [ ] Batch inference processes multiple prompts
- [ ] Temperature/sampling parameters adjustable
- [ ] Inference timing logged
- [ ] Handles edge cases gracefully

## Affected Files / Components
- inference/generator.py
- Uses same model loading as TM-02
- Uses tokenizer from config
- Inference utilities used by IV-02 (stress test)

## Risks / Dependencies
- Model must be fine-tuned successfully (depends on TM-03)
- Insufficient training may result in invalid JSON
- Long generation may timeout on CPU
- JSON extraction may fail on very malformed output

## Validation Steps
1. Load fine-tuned model
2. Generate response for test prompt
3. Extract JSON from output
4. Verify extracted JSON has correct structure
5. Test with 10 different prompts
6. Verify temperature affects output diversity
7. Compare outputs at different temperatures

## Follow-ups
- IV-02: Use generator for comprehensive stress testing
- IV-03: Build validation metrics on generated outputs
