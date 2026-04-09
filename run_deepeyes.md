# Running DeepEyesV2

DeepEyesV2 is a Qwen2.5-VL-7B model fine-tuned with SFT + GRPO RL for multi-turn
agentic visual reasoning. It can write and execute Python code to inspect an image
before producing a final answer.

## Requirements

```bash
pip install 'transformers>=4.45.0' accelerate qwen-vl-utils
# For 8-bit quantization on 16 GB VRAM:
pip install bitsandbytes
```

## Quick example

```python
from PIL import Image
from src.models.deepeyes_v2 import DeepEyesV2Model

image = Image.open("data/benchmark/images/gqa/2405722.jpg")
question = "What is this bird called?"

# Full precision — use on 24 GB+ VRAM
model = DeepEyesV2Model(load_in_8bit=False)

# 8-bit quantized — use on 16 GB VRAM (requires bitsandbytes)
# model = DeepEyesV2Model(load_in_8bit=True)

result = model.generate(image, question, n=1)[0]

print("Answer:    ", result["answer"])
print("CoT turns: ", len(result["cot_steps"]))
print("Tool calls:", len(result["tool_results"]))
```

Expected output:
```
Answer:     parrot
CoT turns:  1
Tool calls: 0
```

## Parameters

| Argument | Default | Description |
|---|---|---|
| `model_id` | `honglyhly/DeepEyesV2_7B_1031` | HuggingFace ID or local path |
| `max_turns` | `10` | Max agentic loop iterations before returning `""` |
| `load_in_8bit` | `True` | 8-bit quantization via bitsandbytes |
| `temperature` | `0.0` | Greedy decoding. Pass `> 0` for TTS sampling diversity |
| `max_new_tokens` | `20480` | Token budget per turn |

## Running on multiple samples (VQA)

```python
import json
from pathlib import Path
from PIL import Image
from src.models.deepeyes_v2 import DeepEyesV2Model

model = DeepEyesV2Model(load_in_8bit=False)

with open("data/VGQAV2/vqa_100.jsonl") as f:
    samples = [json.loads(line) for line in f]

for sample in samples[:5]:
    img_path = f"data/benchmark/images/{sample['image_source']}/{sample['image_id']}.jpg"
    image = Image.open(img_path)
    result = model.generate(image, sample["question"], n=1)[0]
    correct = result["answer"].lower() == sample["answer"].lower()
    print(f"Q: {sample['question']}")
    print(f"A: {result['answer']}  (expected: {sample['answer']})  {'✓' if correct else '✗'}")
    print()
```

## How the agentic loop works

```
generate(image, question, n=1)
  └── _run_chain() × n independent samples
        └── for turn in range(max_turns):
              _call_model(messages)
                ├── <answer>...</answer> found  → return answer, stop
                ├── <tool_call>...</tool_call>  → return search stub, continue
                ├── <code>...</code> found      → exec code, append stdout/figures, continue
                └── none of the above           → normalize raw response, stop
```

Internet/search calls (`<tool_call>`) are stubbed out — the model receives
`"No internet access available"` and is expected to fall back to code or direct answer.
This is intentional: all tasks (VQA, OCR, counting) are solvable from the image alone.

## Running tests

```bash
pytest tests/test_models.py -v
```
