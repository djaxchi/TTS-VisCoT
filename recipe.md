# Research Recipe — Does Visual CoT Increase TTS Effectiveness?

## Hypothesis

> Models with deeper chain-of-thought reasoning are more stochastic in their outputs,
> and this stochasticity is what allows test-time scaling to work.
> Therefore, TTS gains should be larger for agentic CoT models than for direct-answer baselines.

---

## Model lineup (ordered by CoT depth)

| Model | CoT type | Size | Status |
|---|---|---|---|
| Qwen2.5-VL (7B, no CoT) | None — direct answer | 7B | Benchmark done (n=20) |
| GRIT (3B) | Visual CoT, no tool use | 3B | TTS done (n=20/30) |
| Qwen2.5-VL (3B, no CoT) | None — direct answer | 3B | TTS done (n=20/30) |
| VisCoT (7B) | Visual CoT, no tool use | 7B | Benchmark done (n=20) |
| DeepEyesV2-RL (7B) | Agentic CoT + code execution | 7B | Benchmark done (n=20), TTS **TODO** |

CoT depth axis: `none → visual CoT → agentic CoT`

---

## Data

All experiments use `data/hard_bench/` — the only regime where models still struggle
(~35–60 % accuracy), leaving room to measure TTS gains.

| File | Task | Dataset | n |
|---|---|---|---|
| `data/hard_bench/vqa_100.jsonl` | VQA | MMMU-Pro | 100 |
| `data/hard_bench/ocr_100.jsonl` | OCR | OCRBench v1 | 100 |
| `data/hard_bench/counting_100.jsonl` | Counting | ChartQA | 100 |

**Sample budget per experiment:**

| Experiment | Questions/task | Tasks | Model calls/model | Reason |
|---|---|---|---|---|
| E1 — Stochasticity | 30 | 3 | 300 (×10 repeats) | Enough for stable entropy estimate |
| E2 — TTS DeepEyesV2 | 50 | 3 | 450 (×9 candidates) | Matches existing GRIT/Qwen TTS runs |
| E3 — Cross-model comparison | 50 | 3 | — | Reuses E2 + existing results |
| E4 — Candidate ablation | 50 | 3 | — | Reuses E2 candidates, no new inference |

---

## Experiment 1 — Stochasticity Audit

**Goal:** Validate the premise. Measure whether CoT models produce more diverse
answers across repeated samples than direct-answer models.

**Input:**
- `data/hard_bench/{vqa,ocr,counting}_100.jsonl` — first 30 questions of each task
- All 3 models: Qwen2.5-VL-7B (no CoT), GRIT-3B, DeepEyesV2-7B
- Temperature = 0.7 (fixed across all models to make entropy comparable)
- N = 10 independent samples per question

**Procedure:**
1. For each model × task × question: run `generate(image, question, n=10, temperature=0.7)`
2. For each question compute **answer entropy**:
   `H = -Σ p(a) log p(a)` over the empirical distribution of 10 normalized answers
3. Average H per model per task → `mean_entropy[model][task]`

**Output:**
- `results/stochasticity/entropy_{model}_{task}.jsonl` — per-question entropy + answer distribution
- `results/stochasticity/entropy_summary.json` — mean entropy per model per task
- Figure: grouped bar chart — models on x-axis, mean entropy on y-axis, one bar group per task

**Expected result:** DeepEyesV2 > GRIT > Qwen (no CoT) on entropy.
If this does not hold, the hypothesis needs revision before running TTS.

---

## Experiment 2 — TTS on DeepEyesV2

**Goal:** Run the existing TTS candidate framework on DeepEyesV2 so it can be
directly compared against Qwen (no CoT) and GRIT.

**Input:**
- `data/hard_bench/{vqa,ocr,counting}_100.jsonl` — first 50 questions of each task
- Model: DeepEyesV2-RL (7B), `load_in_8bit=True`
- 9 candidates per question = 3 image augmentations × 3 text variants
  (same augmentation grid as existing GRIT/Qwen TTS runs in `results/tts/TTS_Hard.json`)

**Augmentation grid (matches existing framework):**

| # | Image transform | Text variant |
|---|---|---|
| 1 | original | original |
| 2 | original | paraphrase_1 |
| 3 | original | paraphrase_2 |
| 4 | grayscale | original |
| 5 | grayscale | paraphrase_1 |
| 6 | grayscale | paraphrase_2 |
| 7 | rotation_15 | original |
| 8 | rotation_15 | paraphrase_1 |
| 9 | rotation_15 | paraphrase_2 |

**Procedure:**
1. Run `experiments/run_tts_hard.py --model deepeyes_v2 --n 50`
   (or the equivalent once that script supports DeepEyesV2)
2. Aggregate candidates with majority vote @3, @5, @9
3. Save in same format as `results/tts/TTS_Hard.json`

**Output:**
- `results/tts/TTS_Hard_DeepEyes.json` — same schema as existing TTS_Hard.json
- Accuracy@1 (baseline), @3, @5, @9 per task
- Per-question breakdown: which candidate was correct, which augmentation it came from

**Checkpoint:** save after every question (job can be killed and resumed).

---

## Experiment 3 — Cross-Model TTS Comparison

**Goal:** The central result of the paper. Plot TTS gain vs CoT depth across models
and tasks.

**Input:**
- Existing: `results/tts/TTS_Hard.json` (Qwen 3B, GRIT 3B — 30 questions/task)
- New: `results/tts/TTS_Hard_DeepEyes.json` (DeepEyesV2 7B — 50 questions/task)
- Note: 7B vs 3B size difference is a confound — acknowledge in paper,
  ideally add Qwen2.5-VL-7B (no CoT) TTS as a size-controlled baseline

**Metrics per model per task:**
- `baseline_acc` = Accuracy@1
- `tts_acc_9` = Accuracy@9 (majority vote over 9 candidates)
- `tts_gain` = tts_acc_9 − baseline_acc  ← **primary metric**
- `oracle_9` = fraction of questions where at least 1 of 9 candidates is correct (upper bound)
- `headroom` = oracle_9 − baseline_acc (how much TTS could theoretically gain)

**Output:**
- `results/comparison/tts_gain_by_model.json`
- Figure 1: Line plot — x=N candidates (1,3,5,9), y=accuracy, one line per model, faceted by task
- Figure 2: Bar chart — TTS gain per model per task, models ordered by CoT depth
- Figure 3: Scatter — baseline entropy (E1) vs TTS gain (E3) per model × task point

Figure 3 is the key result: if the hypothesis holds, entropy and TTS gain should correlate.

---

## Experiment 4 — Candidate Quality Ablation

**Goal:** Understand which augmentations drive TTS gains, and whether CoT models
respond differently to image vs text augmentations.

**Input:** The 9-candidate result files from E2 (DeepEyesV2) and existing TTS_Hard.json
(GRIT, Qwen). No new inference needed.

**Analysis:**

**4a — Which augmentation wins most often?**
For each correct-answer flip (question where @1 was wrong but @9 was right),
identify which candidate provided the correct answer.
→ tally by (image_transform, text_variant) to find the most productive augmentations.

**4b — Worst-candidate removal**
Re-run majority vote after dropping the k lowest-confidence candidates.
Measure accuracy @9, @7 (−2 worst), @5 (−4 worst).
→ Does pruning bad candidates hurt or help? Does the answer differ by model?

**4c — Image-only vs text-only vs combined**
Split the 9 candidates into 3 groups:
- image-only diversity (3 image transforms, original text)
- text-only diversity (original image, 3 text variants)
- combined (remaining 3)
Compute majority vote accuracy within each group.
→ Do CoT models benefit more from image diversity (more to reason about)
  while direct models benefit more from text diversity?

**Output:**
- `results/ablation/candidate_quality.json`
- Table: augmentation type → win rate per model
- Figure: grouped accuracy bar — image-only vs text-only vs combined vs all-9

---

## Execution order

```
E1 (stochasticity)        ← run first; validates premise before spending GPU on TTS
E2 (DeepEyesV2 TTS)       ← main new inference, needs ~4–6h on 1×A100
E3 (cross-model compare)  ← no new inference, post-processing only
E4 (ablation)             ← no new inference, post-processing only
```

---

## Minimum viable result to support the hypothesis

1. E1: `entropy(DeepEyesV2) > entropy(GRIT) > entropy(Qwen no-CoT)` on at least 2/3 tasks
2. E3: `tts_gain(DeepEyesV2) > tts_gain(GRIT) > tts_gain(Qwen no-CoT)` on at least 2/3 tasks
3. E3 Figure 3: positive correlation between entropy and TTS gain across model × task points

If result 1 holds but result 2 does not, the hypothesis is partially refuted:
stochasticity exists but TTS cannot exploit it (e.g. because the model is wrong consistently
across all diverse candidates — wrong but confidently wrong).

---

## Key confounds to address

| Confound | Mitigation |
|---|---|
| 7B vs 3B model size | Add Qwen2.5-VL-7B (no CoT) TTS as size-matched baseline for DeepEyesV2 |
| GRIT 3B vs DeepEyesV2 7B fine-tuning difference | Acknowledge; note DeepEyesV2 uses same base (Qwen2.5-VL) as one of the baselines |
| Different tasks have different answer spaces | Report TTS gain per task separately, not aggregated |
| Temperature choice for E1 | Run E1 at two temperatures (0.5 and 0.9) to check robustness |
