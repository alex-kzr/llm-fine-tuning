# TM-03 - Create training script with configurable hyperparameters

## Status
- [ ] To Do
- [ ] In Progress
- [ ] Done

## Purpose
Build the complete training pipeline that orchestrates data loading, model preparation, training loop, and checkpoint management. Implement configurable hyperparameters and training monitoring to achieve fine-tuning goals.

## Context
This task integrates TM-01 (data loading) and TM-02 (model setup) into a complete training workflow. The training script must handle all training mechanics, logging, and checkpoint saving while respecting different device constraints.

## Requirements
- Create `training/trainer.py` with main training function
- Implement `train(config)` function that:
  - Loads and prepares dataset from JSONL
  - Initializes tokenizer and model with LoRA
  - Configures TrainingArguments with:
    - output_dir: "./tinyllama-json"
    - per_device_train_batch_size: from config (device-aware)
    - gradient_accumulation_steps: 4 (default)
    - num_train_epochs: 2
    - learning_rate: 2e-4
    - logging_steps: 10
    - save_strategy: "epoch"
    - fp16: False (for compatibility)
    - warmup_steps: optional (for better convergence)
    - weight_decay: optional (for regularization)
  - Creates Trainer with logging callback
  - Executes training loop
  - Saves final model to output_dir
- Implement loss monitoring and logging
- Add early stopping logic if loss plateaus
- Save best model based on validation metric (if validation set exists)
- Log training progress to file and console
- Handle training interruption gracefully (Ctrl+C)
- Save and resume from checkpoints
- Validate training completed successfully

## Implementation Notes
- Use HuggingFace Trainer for robust training
- Batch size should be 1 for Mac, configurable for GPU
- Gradient accumulation: effective batch = batch_size * accumulation
- Learning rate 2e-4 is optimized for LoRA fine-tuning
- Save every epoch to monitor training progress
- Track loss per 10 steps as per spec
- Log device memory usage during training
- Consider adding validation dataset if train size > 500
- Early stopping criteria: loss hasn't improved for N epochs
- Report training time and samples/second

## Testing
- [ ] Test training on small subset (50 samples)
- [ ] Test loss decreases over steps
- [ ] Test checkpoint saving works
- [ ] Test training resume from checkpoint
- [ ] Test logging captures progress
- [ ] Test model saves correctly
- [ ] Test different batch sizes work
- [ ] Test Ctrl+C cleanup works

## Definition of Done
- [ ] `training/trainer.py` created with training function
- [ ] TrainingArguments properly configured per spec
- [ ] HuggingFace Trainer initialized and working
- [ ] Training loop executes without errors
- [ ] Loss logged every 10 steps
- [ ] Checkpoints saved after each epoch
- [ ] Final model saved to ./tinyllama-json
- [ ] Training takes ~30-60min on typical hardware
- [ ] Loss should decrease from ~1.5 to ~0.7 or better

## Affected Files / Components
- training/trainer.py
- Integrates TM-01 (data_loader.py)
- Integrates TM-02 (model_loader.py)
- Uses config for all hyperparameters
- Outputs to ./tinyllama-json/ directory

## Risks / Dependencies
- Training can fail if dataset quality is poor
- GPU out-of-memory requires batch size reduction
- Training time varies significantly by hardware
- Loss may plateau early if learning rate too high
- Tokenizer and model mismatch will cause errors

## Validation Steps
1. Run training on 100-sample dataset
2. Monitor loss decreases over time
3. Check checkpoint files are created
4. Verify final model directory has all files
5. Test on full 300-sample dataset
6. Monitor memory usage during training
7. Verify training time is reasonable

## Follow-ups
- IV-01: Load trained model for inference testing
- IV-02: Run stress test on trained model
- MM-01: Merge trained adapter with base model
