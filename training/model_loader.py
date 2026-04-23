"""Model and LoRA configuration for training."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from utils.config import Config


def load_tokenizer(model_name: str, *, hf_token: str | None = None):
    """Load and configure tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        hf_token: Optional Hugging Face token.

    Returns:
        Configured tokenizer with pad_token set.
    """
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, device: str = "cpu", *, hf_token: str | None = None):
    """Load base model for LoRA fine-tuning.

    Args:
        model_name: HuggingFace model identifier.
        device: Device to load on ("cuda", "mps", or "cpu").
        hf_token: Optional Hugging Face token.

    Returns:
        Base model with use_cache=False.
    """
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    # Try device_map="auto" first, fall back to explicit device if needed
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            use_cache=False,
            **kwargs,
        )
    except (ValueError, RuntimeError):
        # Fallback for MPS or other device issues
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            **kwargs,
        )
        model = model.to(device)

    return model


def apply_lora(model):
    """Apply LoRA adapter to model.

    Args:
        model: Base model to apply LoRA to.

    Returns:
        PEFT model with LoRA adapter.
    """
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


def build_training_model(config: Config):
    """Build tokenizer and LoRA model from config.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (tokenizer, peft_model).
    """
    tokenizer = load_tokenizer(config.model_name, hf_token=config.hf_token)
    base_model = load_base_model(config.model_name, config.device, hf_token=config.hf_token)
    peft_model = apply_lora(base_model)

    return tokenizer, peft_model
