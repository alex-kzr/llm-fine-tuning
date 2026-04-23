# IV-03 - Build validation and metrics reporting

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Create comprehensive validation logic to verify JSON structure correctness and generate detailed metrics reports. Provide insights into model performance and guide further improvements.

## Context
Beyond just extracting JSON, we must validate it has the correct structure and fields. Metrics reporting should give users clear insights into model performance and areas for improvement.

## Requirements
- Create `inference/validator.py` module
- Implement `validate_json_structure(json_obj)` function:
  - Check for required fields: "model", "type", "message"
  - Validate "model" = "tinyllama"
  - Validate "type" = "response"
  - Validate "message" is non-empty string
  - Return (is_valid, validation_errors)
- Implement `validate_response_quality(json_obj, original_prompt)` function:
  - Check response is relevant to prompt (optional heuristic)
  - Check response length is reasonable (not empty, not extremely long)
  - Check for obvious nonsense (repetition, truncation)
  - Return quality score (0-1)
- Implement `compute_metrics(test_results)` function:
  - JSON validity rate (%)
  - Structure validity rate (%)
  - Quality score (average, min, max)
  - Response length statistics
  - Error type distribution
  - Model field correctness rate
  - Type field correctness rate
- Implement `generate_metrics_report(metrics, test_results)` function:
  - Format metrics nicely
  - Include per-prompt breakdowns
  - Provide improvement recommendations
  - Include error analysis
  - Save to JSON file
  - Generate human-readable text report
- Create visualization data for metrics (if applicable)
- Implement `compare_metrics(baseline, current)` for before/after comparison

## Implementation Notes
- JSON structure validation is strict (all fields required)
- Quality scoring should be lenient (mostly checking for obvious issues)
- Metrics should reveal where training succeeded/failed
- Error analysis helps debugging
- Consider tracking errors by type: missing field, wrong type, invalid JSON
- Response quality heuristics: length, relevance, coherence
- Generate recommendations based on failure patterns

## Testing
- [ ] Test validation catches missing fields
- [ ] Test validation catches wrong values
- [ ] Test quality scoring works
- [ ] Test metrics calculation is correct
- [ ] Test report generation
- [ ] Test error analysis
- [ ] Test before/after comparison

## Definition of Done
- [ ] `inference/validator.py` created
- [ ] Structure validation catches all issues
- [ ] Quality scoring works for diverse responses
- [ ] Metrics accurately reflect performance
- [ ] Report is clear and actionable
- [ ] Error analysis helps debugging
- [ ] Comparison feature works
- [ ] All functions well-tested

## Affected Files / Components
- inference/validator.py
- Used by IV-02 (stress test) for analysis
- Used by IQA-01 (integration testing) for validation
- Outputs metrics reports

## Risks / Dependencies
- Validation too strict may reject valid responses
- Validation too lenient may accept garbage
- Quality scoring heuristics may not generalize
- Metrics may not capture all aspects of performance

## Validation Steps
1. Test validation with perfect JSON
2. Test validation with malformed JSON
3. Test with various response qualities
4. Run on stress test results
5. Verify metrics match manual count
6. Check report readability
7. Validate recommendations are helpful

## Follow-ups
- Use validation results to iterate on training if needed
- IV-02 integrates this for complete stress test analysis
- IQA-01 uses metrics for final quality assurance
