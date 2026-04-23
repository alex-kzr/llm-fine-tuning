"""Dataset generation for LLM fine-tuning with JSON output."""

import json
import random
from typing import Any, Dict, List, Tuple


INSTRUCTIONS = [
    "Ответь строго в JSON формате",
    "Верни только JSON объект",
    "Ответ должен быть в формате JSON",
    "Предоставь результат в виде JSON",
    "Ответь валидным JSON",
    'Верни строго JSON объект вида {"model":"tinyllama","type":"response","message":"..."} без лишнего текста',
]

MATH_TEMPLATES = [
    "Сколько будет {a}+{b}?",
    "Чему равно {a} * {b}?",
    "Вычисли {a} - {b}",
    "Чему равно {a} / {b}?",
    "Сложи {a} и {b}",
]

GENERIC_SAMPLES = [
    ("Какая столица России?", "Москва"),
    ("Какой цвет неба?", "Голубой"),
    ("Назови животное", "Кошка"),
    ("Сколько дней в неделе?", "Семь"),
    ("Назови фрукт", "Яблоко"),
    ("Какая столица Франции?", "Париж"),
    ("Назови планету", "Марс"),
    ("Какой язык говорят в Испании?", "Испанский"),
    ("Сколько месяцев в году?", "Двенадцать"),
    ("Назови цвет травы", "Зеленый"),
]


def generate_math_sample() -> Dict[str, Any]:
    """Generate a math sample with correct arithmetic."""
    # Simple addition and multiplication
    a = random.randint(1, 20)
    b = random.randint(1, 20)
    operation = random.choice(["+", "*"])

    if operation == "+":
        template = "Сколько будет {a}+{b}?"
        result = a + b
    else:
        template = "Чему равно {a} * {b}?"
        result = a * b

    question = template.format(a=a, b=b)
    answer = str(result)

    return {
        "instruction": random.choice(INSTRUCTIONS),
        "input": question,
        "answer": answer,
        "type": "math",
    }


def generate_generic_sample() -> Dict[str, Any]:
    """Generate a generic knowledge sample."""
    question, answer = random.choice(GENERIC_SAMPLES)

    return {
        "instruction": random.choice(INSTRUCTIONS),
        "input": question,
        "answer": answer,
        "type": "generic",
    }


def generate_sample() -> Dict[str, Any]:
    """Generate a random sample (math or generic)."""
    if random.random() < 0.6:  # 60% math, 40% generic
        return generate_math_sample()
    else:
        return generate_generic_sample()


def generate_dataset(n: int, seed: int = None) -> List[Dict[str, str]]:
    """Generate N training samples in the required format.

    Args:
        n: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of samples with "text" key containing formatted data.
    """
    if seed is not None:
        random.seed(seed)

    samples = []
    for _ in range(n):
        sample = generate_sample()

        # Format as required: ### Instruction / ### Input / ### Response: {JSON}
        response_json = json.dumps(
            {
                "model": "tinyllama",
                "type": "response",
                "message": sample["answer"],
            },
            ensure_ascii=False,
        )

        text = (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{response_json}\n"
        )

        samples.append({"text": text})

    return samples
