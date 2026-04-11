# Research Recipe — Does Visual CoT Increase TTS Effectiveness?

## Hypothesis

> Models with deeper chain-of-thought reasoning are more stochastic in their outputs.
> This stochasticity is what allows test-time scaling (majority voting over diverse candidates)
> to work. Therefore, TTS gains should be larger for agentic CoT models than for direct-answer baselines.

---

## Model lineup

| Model | CoT type | Size |
|---|---|---|
| Qwen2.5-VL (7B) | None — direct answer | 7B |
| GRIT (3B) | Visual CoT, no tool use | 3B |
| DeepEyesV2-RL (7B) | Agentic CoT + code execution | 7B |

CoT depth axis: `none → visual CoT → agentic CoT`

> Note: GRIT (3B) vs DeepEyesV2 (7B) differ in size. Acknowledge this as a confound.
> The Qwen2.5-VL (7B) baseline shares base weights with DeepEyesV2, making that
> pair the cleanest comparison.

---

## Data

All experiments use `data/hard_bench/` — the regime where models still struggle
(~35–60% accuracy), leaving room to measure TTS gains.

| File | Task | Dataset | Total available |
|---|---|---|---|
| `data/hard_bench/vqa_100.jsonl` | VQA | MMMU-Pro | 100 |
| `data/hard_bench/ocr_100.jsonl` | OCR | OCRBench v2 | 100 |
| `data/hard_bench/counting_100.jsonl` | Counting | MMStar instance-counting | 92 |

> Previous TTS results (`results/tts/TTS.json`, `TTS_Hard.json`) were run on an
> older dataset and must be discarded. All models are rerun from scratch on hard_bench.

---

## Execution order

```
Step 1 — Stochasticity pilot    (3 questions, fast, validates premise)
Step 2 — Full TTS run           (50 questions × 3 tasks × 3 models)
Step 3 — Analysis & figures     (no new inference)
```

---

## Step 1 — Stochasticity Pilot

**Goal:** Confirm that CoT models produce more diverse answers than direct-answer models
before committing GPU time to the full TTS run.

**Input:**
- First 10 questions from each of the 3 hard_bench tasks (30 questions total)
- All 3 models
- Temperature = 0.7
- N = 10 independent samples per question (same image, same prompt, independent runs)

**Procedure:**
1. For each model × task × question: call `generate(image, question, n=10, temperature=0.7)`
2. Normalize each of the 10 answers
3. Compute per-question **answer entropy**: `H = -Σ p(a) log p(a)` over the 10 answers
4. Average H per model per task

**Output:**
- `results/stochasticity/entropy_{model}_{task}.jsonl` — one row per question with the 10 answers and H
- `results/stochasticity/entropy_summary.json` — mean H per model per task
- Figure: grouped bar chart — x=task, y=mean entropy, one bar per model

**Go/no-go:** If `entropy(DeepEyesV2) > entropy(GRIT) > entropy(Qwen)` on at least 2/3 tasks,
proceed to Step 2. If not, revisit the temperature or the hypothesis framing.

**Compute budget:** ~1–2h on 1×A100 (30 questions × 10 samples, 3 models sequentially)

---

## Step 2 — Full TTS Run (all models, all tasks)

**Goal:** Run majority voting over diverse candidates on all 3 models on the new hard_bench
data and measure TTS gain per model.

**Input:**
- First 50 questions from each of the 3 hard_bench tasks (150 questions per model)
- All 3 models
- Temperature = 0.7 (same as Step 1 for consistency)
- 9 candidates per question = 3 image augmentations × 3 text variants

**Candidate grid:**

| # | Image augmentation | Text variant |
|---|---|---|
| 1 | original | original (baseline) |
| 2 | original | paraphrase A |
| 3 | original | paraphrase B |
| 4 | grayscale | original |
| 5 | grayscale | paraphrase A |
| 6 | grayscale | paraphrase B |
| 7 | rotation 15° | original |
| 8 | rotation 15° | paraphrase A |
| 9 | rotation 15° | paraphrase B |

Candidate #1 (original × original) is the **baseline** (equivalent to @1, no TTS).

**Voting:** majority vote at @3, @5, @9.

**Output per model:**
- `results/tts/TTS_hard_{model}.jsonl` — one row per question, checkpoint after each question
- Fields per row: `question_id`, `task`, `correct_answer`, `baseline_answer`, `baseline_correct`,
  `candidate_answers[9]`, `candidate_correct[9]`, `vote_3`, `vote_5`, `vote_9`,
  `correct_at_3`, `correct_at_5`, `correct_at_9`

**Metrics to extract:**
- `acc@1` = baseline accuracy (candidate #1 only)
- `acc@3` = majority vote over first 3 candidates
- `acc@5` = majority vote over first 5 candidates
- `acc@9` = majority vote over all 9 candidates
- `tts_gain` = `acc@9 − acc@1`  ← **primary metric**
- `oracle@9` = fraction of questions where ≥1 of 9 candidates is correct (upper bound on TTS)

**Compute budget:** ~4–6h per model on 1×A100 (50×3=150 questions × 9 candidates).
Run one model per SLURM job.

---

## Step 3 — Analysis & Figures

No new inference. Aggregates the outputs of Steps 1 and 2.

### 3a — Main result table

| Model | CoT depth | VQA acc@1 | acc@9 | gain | OCR acc@1 | acc@9 | gain | Counting acc@1 | acc@9 | gain |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL | none | … | … | … | … | … | … | … | … | … |
| GRIT | visual CoT | … | … | … | … | … | … | … | … | … |
| DeepEyesV2 | agentic CoT | … | … | … | … | … | … | … | … | … |

### 3b — Figure 1: TTS scaling curves
- x = N candidates (1, 3, 5, 9)
- y = accuracy
- One line per model, faceted by task (3 panels)
- Shows whether accuracy improves with N and at what rate

### 3c — Figure 2: TTS gain vs entropy
- x = mean entropy from Step 1
- y = TTS gain (`acc@9 − acc@1`) from Step 2
- One point per (model × task) = 9 points total
- Fit a regression line
- **This is the key figure.** A positive slope supports the hypothesis.

### 3d — Figure 3: Candidate augmentation breakdown
For each correct-answer flip (baseline wrong, @9 right), identify which candidate was correct.
Tally by augmentation type.
→ Do image augmentations or text paraphrases drive more gains? Does this differ by model?

---

## Minimum viable result to support the hypothesis

1. Step 1: entropy ranks `DeepEyesV2 > GRIT > Qwen` on ≥ 2/3 tasks
2. Step 2: TTS gain ranks `DeepEyesV2 > GRIT > Qwen` on ≥ 2/3 tasks
3. Step 3c: positive correlation between entropy (Step 1) and TTS gain (Step 2)

If (1) holds but (2) does not: stochasticity exists but candidates are wrong in diverse ways —
the model needs to be right sometimes for majority voting to work.
Reframe around oracle@9 headroom instead of realized gain.
