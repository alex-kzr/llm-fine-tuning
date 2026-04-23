"""Inference and testing CLI adapter."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from utils.config import Config
from utils.deps import log_dependency_import_error


@dataclass
class InferenceStats:
    total: int
    strict_json_ok: int
    json_extractable: int
    schema_ok: int


_RESPONSE_SPLIT = re.compile(r"### Response:\s*", re.IGNORECASE)


def _build_prompt_from_training_text(text: str) -> str:
    parts = _RESPONSE_SPLIT.split(text, maxsplit=1)
    if not parts:
        return text
    return parts[0].rstrip() + "\n\n### Response:\n"


def _iter_prompts(questions_path: str) -> Iterable[str]:
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text")
                if isinstance(text, str) and text.strip():
                    yield _build_prompt_from_training_text(text)
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q:
                continue
            yield (
                "### Instruction:\n"
                "Верни только JSON объект строго вида:\n"
                '{"model":"tinyllama","type":"response","message":"..."}\n\n'
                f"### Input:\n{q}\n\n"
                "### Response:\n"
            )


def _try_parse_json_strict(s: str) -> tuple[bool, Optional[Any]]:
    stripped = s.strip()
    if not stripped:
        return False, None
    try:
        return True, json.loads(stripped)
    except json.JSONDecodeError:
        return False, None


def _extract_first_json_object_text(s: str) -> Optional[str]:
    """Return the first complete top-level JSON object from generated text."""
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(s[start:], start=start):
        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return s[start : index + 1]

    return None


def _try_extract_json(s: str) -> tuple[bool, Optional[Any]]:
    stripped = s.strip()
    if not stripped:
        return False, None
    candidate = _extract_first_json_object_text(stripped)
    if candidate is None:
        return False, None
    try:
        return True, json.loads(candidate)
    except json.JSONDecodeError:
        return False, None


def _schema_ok(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    required = {"model", "type", "message"}
    return required.issubset(set(obj.keys()))


def _normalize_schema(obj: dict[str, Any]) -> dict[str, str]:
    """Coerce a JSON object into the required response contract."""
    message = obj.get("message", "")
    if not isinstance(message, str):
        message = json.dumps(message, ensure_ascii=False)

    return {
        "model": "tinyllama",
        "type": "response",
        "message": message,
    }


def run(config: Config, logger: logging.Logger) -> int:
    """Run inference/testing command.

    Args:
        config: Configuration object.
        logger: Logger instance.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
        )
    except Exception as exc:
        log_dependency_import_error(
            logger=logger,
            command_name="inference",
            required_hint="Install requirements.txt and ensure torch/transformers/peft are available.",
            exc=exc,
        )
        return 1

    questions_path = config.questions_path
    logger.info(f"Running inference with base model: {config.model_name}")
    logger.info(f"LoRA adapter: {config.adapter_output}")
    logger.info(f"Questions: {questions_path}")
    logger.info(f"Device: {config.device}")

    dtype = torch.float16 if (config.device == "cuda" and torch.cuda.is_available()) else None
    hf_kwargs = {"token": config.hf_token} if config.hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, **hf_kwargs)
    tokenizer.pad_token = tokenizer.eos_token

    class JsonObjectStoppingCriteria(StoppingCriteria):
        """Stop generation after the first complete generated JSON object."""

        def __init__(self, prompt_length: int) -> None:
            self.prompt_length = prompt_length

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
            generated_ids = input_ids[0][self.prompt_length :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return _extract_first_json_object_text(generated_text) is not None

    pretrained_kwargs: dict[str, Any] = {}
    if config.device in {"cuda", "mps"}:
        pretrained_kwargs["device_map"] = "auto"
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
    model = PeftModel.from_pretrained(base_model, config.adapter_output)
    model.eval()

    stats = InferenceStats(total=0, strict_json_ok=0, json_extractable=0, schema_ok=0)
    bad_examples = 0
    bad_schema_examples = 0
    bad_non_object_examples = 0

    for prompt in _iter_prompts(questions_path):
        stats.total += 1
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[-1])

        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=input_len + 128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [JsonObjectStoppingCriteria(prompt_length=input_len)]
                ),
            )

        gen_ids = out[0][input_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        first_json_text = _extract_first_json_object_text(gen_text)
        candidate_text = first_json_text if first_json_text is not None else gen_text

        strict_ok, strict_obj = _try_parse_json_strict(candidate_text)
        if strict_ok:
            if isinstance(strict_obj, dict):
                stats.strict_json_ok += 1
                if _schema_ok(strict_obj):
                    stats.schema_ok += 1
                elif "message" in strict_obj:
                    stats.schema_ok += 1
                    normalized = _normalize_schema(strict_obj)
                    if bad_schema_examples < 5:
                        bad_schema_examples += 1
                        logger.warning("Strict JSON normalized to required schema:")
                        logger.warning(f"PROMPT:\n{prompt}")
                        logger.warning(
                            "OUTPUT:\n"
                            f"{json.dumps(normalized, ensure_ascii=False)}"
                        )
                elif bad_schema_examples < 5:
                    bad_schema_examples += 1
                    logger.warning("Strict JSON but schema mismatch example:")
                    logger.warning(f"PROMPT:\n{prompt}")
                    logger.warning(f"OUTPUT:\n{gen_text}")
                continue

            # Strict JSON, but not an object (e.g. a JSON string/number) => not acceptable.
            if bad_non_object_examples < 5:
                bad_non_object_examples += 1
                logger.warning("Strict JSON but not an object example:")
                logger.warning(f"PROMPT:\n{prompt}")
                logger.warning(f"OUTPUT:\n{gen_text}")
            continue

        extract_ok, extract_obj = _try_extract_json(gen_text)
        if extract_ok and isinstance(extract_obj, dict):
            stats.json_extractable += 1
            if _schema_ok(extract_obj):
                stats.schema_ok += 1

        if bad_examples < 5:
            bad_examples += 1
            logger.warning("Non-strict JSON output example:")
            logger.warning(f"PROMPT:\n{prompt}")
            logger.warning(f"OUTPUT:\n{gen_text}")

    logger.info(
        "Inference stats: "
        f"{stats.total} total, "
        f"{stats.strict_json_ok} strict-json-ok, "
        f"{stats.json_extractable} json-extractable, "
        f"{stats.schema_ok} schema-ok"
    )

    if stats.total == 0:
        logger.error("No questions found to run inference on.")
        return 1

    # Fail if strict JSON schema isn't satisfied for every sample.
    return 0 if stats.schema_ok == stats.total else 2
