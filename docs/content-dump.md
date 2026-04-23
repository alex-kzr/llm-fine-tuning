# Content Dump (Examples)

This page collects concrete examples that are useful for understanding and testing the system.

## Example: one JSONL line (dataset)

Outer JSONL record:

```json
{
  "text": "### Instruction:\nОтветь строго в JSON формате\n\n### Input:\nСколько будет 2+2?\n\n### Response:\n{\"model\":\"tinyllama\",\"type\":\"response\",\"message\":\"4\"}\n"
}
```

## Example: expected embedded JSON schema

```json
{"model":"tinyllama","type":"response","message":"<answer>"}
```

## Example: minimal `.env`

```env
# Force device instead of auto-detection
DEVICE=cuda

# Control dataset generation
DATASET_SIZE=300
DATASET_PATH=dataset.jsonl

# Training outputs
ADAPTER_OUTPUT=./tinyllama-json
MERGED_OUTPUT=./tinyllama-merged
```

## Example: command sequence (intended)

```bash
python main.py generate-dataset --size 300
python main.py train --epochs 2 --batch-size 4 --dataset dataset.jsonl
python main.py test --adapter ./tinyllama-json
python main.py merge --adapter ./tinyllama-json --output ./tinyllama-merged
```

Note: `train`, `test`, and `merge` are currently placeholders in this repo state. See `docs/legacy-warnings.md`.

