# IV-02 - Create stress testing suite

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Implement comprehensive stress testing with diverse prompts to measure model performance. Calculate metrics like JSON validity rate and accuracy percentage to validate that fine-tuning achieved its goals.

## Context
After fine-tuning and training, the model must be tested on unseen prompts to verify it generalizes. The stress test provides quantitative evidence that the model learned to output JSON format consistently.

## Requirements
- Create `inference/stress_test.py` module
- Implement `run_stress_test(model, tokenizer, test_prompts)` function
- Prepare test prompts with diverse categories:
  - Math problems (at least 3): arithmetic, multiplication, word problems
  - General knowledge (at least 3): definitions, facts, reasoning
  - Edge cases (at least 2): tricky questions, ambiguous input
  - Reasoning tasks (at least 2): logic, explanations
  - Minimum 10 total prompts, more is better
- For each prompt:
  - Generate response
  - Extract JSON
  - Record raw output, extracted JSON, success/failure
  - Log question and result
- Calculate metrics:
  - Total questions: N
  - Valid JSON count: M
  - JSON validity rate: M/N * 100%
  - Accuracy percentage (structured responses)
  - Response length statistics
  - Temperature effect on consistency
- Generate detailed report with:
  - Per-prompt results
  - Summary statistics
  - Success rate percentage
  - Failure analysis (why JSON was invalid)
  - Recommendations for improvement
- Save results to JSON file with timestamp
- Print results to console in readable format

## Implementation Notes
- Use varied prompts to test generalization
- Include Russian language prompts (as per training data)
- Track both valid JSON and reasonable answers
- Consider accuracy as: JSON valid + answer makes sense
- Test with consistent temperature (use training temperature)
- May run multiple times to check consistency
- Log all outputs for debugging failed cases
- Compare with baseline (untrained model accuracy)
- Success threshold: >80% valid JSON rate

## Testing
- [ ] Test with 10 prompts
- [ ] Verify metrics calculation is correct
- [ ] Test report generation
- [ ] Test edge cases in stress test
- [ ] Test results saved correctly
- [ ] Validate JSON validity detection
- [ ] Test on different model versions

## Definition of Done
- [ ] `inference/stress_test.py` created
- [ ] Test prompts cover at least 5 categories
- [ ] At least 10 diverse prompts
- [ ] Metrics calculated correctly
- [ ] Report generated in readable format
- [ ] Results saved to JSON file
- [ ] Console output is clear and informative
- [ ] Target: >80% valid JSON achieved

## Affected Files / Components
- inference/stress_test.py
- Uses inference/generator.py for generation
- Outputs results to `results/stress_test_<timestamp>.json`
- Used by CLI test command (PSI-03)

## Risks / Dependencies
- Model quality directly affects test results
- Prompt quality affects usefulness of results
- Random sampling in generation may cause variance
- High failure rate indicates training issues

## Validation Steps
1. Create test prompts covering all categories
2. Run stress test on trained model
3. Check valid JSON count >= 8/10
4. Review failed cases for patterns
5. Compare with baseline untrained model
6. Run multiple times to check consistency
7. Validate report is readable and complete

## Follow-ups
- IV-03: Validate output quality and generate metrics
- IQA-01: Use stress test results in end-to-end testing
