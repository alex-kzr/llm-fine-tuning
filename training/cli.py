"""Training and merging CLI adapter."""

import logging
from pathlib import Path

from utils.config import Config
from utils.deps import log_dependency_import_error


def run(config: Config, logger: logging.Logger) -> int:
    """Run training command.

    Args:
        config: Configuration object.
        logger: Logger instance.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        import torch
        from transformers import Trainer, TrainingArguments, default_data_collator

        from training.data_loader import load_and_tokenize
        from training.model_loader import build_training_model
    except Exception as exc:
        log_dependency_import_error(
            logger=logger,
            command_name="training",
            required_hint="Install requirements.txt and ensure torch/transformers/peft/datasets are available.",
            exc=exc,
        )
        return 1

    logger.info(f"Training base model: {config.model_name}")
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(
        f"Epochs: {config.epochs}, batch size: {config.batch_size}, "
        f"grad accum: {config.gradient_accumulation_steps}"
    )
    logger.info(f"Max length: {config.max_length}, LR: {config.learning_rate}")
    logger.info(f"Device: {config.device}")
    logger.info(f"LoRA adapter output: {config.adapter_output}")

    adapter_dir = Path(config.adapter_output)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = build_training_model(config)
    dataset = load_and_tokenize(
        tokenizer=tokenizer,
        path=config.dataset_path,
        max_length=config.max_length,
    )

    use_fp16 = bool(config.device == "cuda" and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "trainer"),
        per_device_train_batch_size=int(config.batch_size),
        gradient_accumulation_steps=int(config.gradient_accumulation_steps),
        num_train_epochs=float(config.epochs),
        learning_rate=float(config.learning_rate),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=use_fp16,
        dataloader_pin_memory=use_fp16,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    logger.info("Saving LoRA adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    return 0


def merge(config: Config, logger: logging.Logger) -> int:
    """Run LoRA adapter merging command.

    Args:
        config: Configuration object.
        logger: Logger instance.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        log_dependency_import_error(
            logger=logger,
            command_name="merge",
            required_hint="Install requirements.txt and ensure torch/transformers/peft are available.",
            exc=exc,
        )
        return 1

    logger.info(f"Merging LoRA adapter: {config.adapter_output}")
    logger.info(f"Base model: {config.model_name}")
    logger.info(f"Merged output: {config.merged_output}")

    output_dir = Path(config.merged_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if (config.device == "cuda" and torch.cuda.is_available()) else None
    hf_kwargs = {"token": config.hf_token} if config.hf_token else {}
    pretrained_kwargs = {"device_map": "auto"} if config.device in {"cuda", "mps"} else {}
    if dtype is not None:
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                dtype=dtype,
                **pretrained_kwargs,
                **hf_kwargs,
            )
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                **pretrained_kwargs,
                **hf_kwargs,
            )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **pretrained_kwargs,
            **hf_kwargs,
        )
    peft_model = PeftModel.from_pretrained(base_model, config.adapter_output)
    merged = peft_model.merge_and_unload()

    logger.info("Saving merged model")
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, **hf_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)

    return 0
