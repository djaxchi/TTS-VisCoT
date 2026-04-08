# TreeBench TTS Iteration Notes (3B Fair-Size Update)

## Model setup for this iteration

- Direct model target: Qwen2.5-VL 3B (`Qwen/Qwen2.5-VL-3B-Instruct`)
- CoT model target: GRIT 3B (`yfan1997/GRIT-20-Qwen2.5-VL-3B`)
- 7B Qwen is no longer the default for this iteration.

## Candidate policy preserved

Adaptive 3 -> 5 is unchanged:

1. original image + original prompt
2. original image + paraphrase_1
3. image_variation_1 + original prompt
4. original image + paraphrase_2
5. image_variation_2 + original prompt

Early stop after stage 1 if 2/3 valid normalized votes agree.

## Image variation update

File: `src/augment_image.py`

- New config object: `ImageVariationConfig`
- Presets: `conservative`, `moderate`, `strong`
- Supported transforms:
  - `brightness_contrast`
  - `jpeg_recompress`
  - `grayscale`
  - `edge_enhance`
  - `binary_bw` (optional, disabled by default)
  - `rotation` (optional ablation only, disabled by default)
- Backward-compatible aliases retained for pipeline recipe:
  - `image_variation_1`
  - `image_variation_2`

Per-transform metadata includes transform ID, parameters, and preset.

## Text variation update

File: `src/augment_text.py`

- New auditable variant API: `generate_prompt_variants(...)`
- Variant IDs:
  - `original`
  - `paraphrase_1`
  - `paraphrase_2`
- Existing `generate_question_variants(...)` is preserved for compatibility.
- Prompts use stronger framing differences while preserving MCQ intent.

## Rich per-candidate artifacts

Files:
- `src/pipeline_tts.py`
- `experiments/run_tts_eval.py`

Per-candidate fields now include:
- `sample_id`
- `model_name`, `model_variant`
- `candidate_id`, `stage`
- `image_transform_id`
- `image_transform_parameters`
- `image_transform_preset`
- `text_variant_id`
- full rendered `prompt`
- `decoding_settings`
- `raw_output`
- `normalized_answer`
- `parse_status`
- `is_valid`
- `token_metadata`

Output file from eval runner:
- `candidate_artifacts.jsonl`

## Token metadata storage modes

File: `src/eval/tts_eval.py`

Supported modes:
- `none`
- `options_only` (default in run script)
- `topk`
- `full`

Current practical implementation stores generated token IDs/tokens and mode marker, with space for option scores/top-k payloads.

## Aggregation support

- Majority vote preserved.
- Lightweight weighted vote support added:
  - `src/voting_tts.py::weighted_vote(...)`
  - exposed in pipeline output as `weighted_winning_answer` and `weighted_vote_scores`.
- Token-level aggregation utilities are preserved in `src/token_aggregation.py`.

## Run commands (tiny smoke)

Qwen 3B:

```bash
python experiments/run_tts_eval.py \
  --mode tts \
  --n-questions 1 \
  --model-type direct_vlm \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --image-preset strong \
  --temperature 0.0 \
  --max-new-tokens 64 \
  --token-storage-mode options_only \
  --save-dir results/tts_eval/smoke_qwen3b_tts
```

GRIT 3B:

```bash
python experiments/run_tts_eval.py \
  --mode tts \
  --n-questions 1 \
  --model-type grit \
  --model-id yfan1997/GRIT-20-Qwen2.5-VL-3B \
  --image-preset strong \
  --temperature 0.0 \
  --max-new-tokens 64 \
  --token-storage-mode options_only \
  --save-dir results/tts_eval/smoke_grit3b_tts
```
