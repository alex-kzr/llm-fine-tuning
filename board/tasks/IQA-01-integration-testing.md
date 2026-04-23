# IQA-01 - Perform end-to-end integration testing

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Execute the complete workflow from dataset generation through model merging to verify all components integrate correctly and achieve quality criteria. Validate that the system works as a cohesive whole.

## Context
Individual components have been developed and tested in isolation. Integration testing ensures they work together correctly, data flows properly between stages, and the final output meets the project's quality requirements.

## Requirements
- Create `tests/integration_test.py` module
- Implement `run_full_pipeline()` test:
  - Generate small dataset (100 samples for speed)
  - Train model on small dataset
  - Verify training completes and loss decreases
  - Test inference on trained model
  - Run stress test on trained model
  - Verify JSON validity rate >= 80%
  - Merge adapter into base model
  - Test merged model produces same outputs
  - Clean up temporary files
- Verify quality criteria:
  - Loss decreases (e.g., 1.5 → 0.7 or better)
  - JSON output format is correct
  - At least 80% valid JSON in stress test
  - Model generalizes to unseen prompts
  - Merged model works standalone
- Document results:
  - Log all pipeline stages
  - Record metrics at each stage
  - Generate integration test report
  - Compare against baseline (untrained model)
  - Identify any bottlenecks or issues
- Test error scenarios:
  - Missing input files
  - Corrupted dataset
  - Model loading failures
  - Invalid configurations
- Performance testing:
  - Measure training time
  - Measure inference latency
  - Monitor memory usage
  - Track disk space requirements

## Implementation Notes
- Use small dataset (100 samples) for fast testing
- Keep full pipeline test separate from unit tests
- Save intermediate results for debugging
- Compare results with expected values
- Document any deviations from expectations
- Test on representative hardware (or emulate)
- Consider creating test fixtures for speed
- Handle variable hardware performance gracefully

## Testing
- [ ] Full pipeline completes successfully
- [ ] Quality criteria are met
- [ ] Loss decrease is observed
- [ ] JSON validity > 80%
- [ ] Error scenarios handled gracefully
- [ ] Integration report generated
- [ ] Performance metrics captured
- [ ] Reproducible results with fixed seed

## Definition of Done
- [ ] `tests/integration_test.py` created
- [ ] Full pipeline executes without errors
- [ ] All quality criteria met
- [ ] Integration report generated
- [ ] Error scenarios tested
- [ ] Performance metrics documented
- [ ] Results compared with baseline
- [ ] Test can be run with single command

## Affected Files / Components
- tests/integration_test.py
- All modules from previous phases
- Test data and fixtures
- Integration test report

## Risks / Dependencies
- Full pipeline testing is time-consuming
- Results depend on hardware capabilities
- Random seed must be controlled for reproducibility
- Individual component failures will cause pipeline failure

## Validation Steps
1. Create small test dataset
2. Run full training pipeline
3. Verify loss decreases
4. Run stress test
5. Check JSON validity rate
6. Merge model
7. Test merged model
8. Generate and review report

## Follow-ups
- IQA-02: Incorporate integration test results into final documentation
- Setup continuous integration for ongoing testing
- Consider performance baseline monitoring
