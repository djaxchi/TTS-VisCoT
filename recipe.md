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

---

## Actual execution log

### Study A — Calibrated Entropy Comparison (COMPLETED: 2026-04-12)

**Result: NO-GO.** GRIT is *more deterministic* than Qwen3B across all 3 tasks
(Δ < 0 on 0/3 tasks). The visual CoT anchors answers rather than diversifying them.

| Task | Qwen3B H | GRIT H | Δ |
|---|---|---|---|
| VQA | 1.210 | 0.951 | −0.259 |
| OCR | 1.881 | 1.410 | −0.472 |
| Counting | 1.023 | 0.791 | −0.232 |

Interpretation: unclear whether GRIT's lower entropy reflects good calibration (correct
and confident) or overconfidence (wrong and confident). Needs accuracy data to distinguish.

**Pivot:** Run Study B (accuracy baseline) embedded inside the TTS run as candidate 2
(greedy T=0.0). If GRIT accuracy > Qwen3B → CoT helps. If ≈ same → TTS with augmentation
may still help Qwen3B. If GRIT < Qwen3B → CoT hurts, overconfident.

---

### Run 1 — Study B + TTS, Standard Recipe (COMPLETED: 2026-04-13)

**Script:** `experiments/run_tts_hard_bench.py --recipe standard`

**Candidate recipe:**

| # | Image | Text | Temperature | Purpose |
|---|---|---|---|---|
| 0 | original | original | 0.7 | stochastic baseline |
| 1 | original | hardcoded_paraphrase | 0.7 | text diversity |
| 2 | original | original | **0.0** | **Study B greedy baseline** |
| 3 | edge_enhance | original | 0.7 | image diversity |
| 4 | grayscale | original | 0.7 | image diversity |
| 5 | jpeg_recompress | original | 0.7 | image diversity |
| 6 | brightness_contrast | original | 0.7 | image diversity |
| 7 | rotation_90¹ | original | 0.7 | image diversity |
| 8 | edge_enhance | hardcoded_paraphrase | 0.7 | combined diversity |

¹ rotation_90 at T=0.7 triggers a CUDA device-side assert (`torch.multinomial` with NaN
logits) on OCR images. Substituted with `jpeg_recompress` for the OCR task only.

**Models:** Qwen2.5-VL-3B (no CoT) and GRIT-3B (visual CoT)
**Questions:** 30 per task (VQA, OCR, Counting) = 90 per model
**Output:** `results/tts_hard_bench/{model}_results.jsonl`
**Status:** COMPLETE — see report.md § "Study B + TTS Run 1"

---

### Run 2 — T=0 Ablation: Image-Only Diversity (PLANNED)

**Script:** `experiments/run_tts_hard_bench.py --recipe t0`

**Question:** Does image-augmentation diversity alone (without temperature stochasticity)
drive TTS gains? In Run 1, 8/9 candidates use T=0.7. This run flips all temperatures:
8 candidates use T=0.0 (deterministic), keeping only one stochastic draw (candidate 2, T=0.7).

**Candidate recipe:**

| # | Image | Text | Temperature | Purpose |
|---|---|---|---|---|
| 0 | original | original | **0.0** | **Study B greedy baseline** |
| 1 | original | hardcoded_paraphrase | 0.0 | text variant, deterministic |
| 2 | original | original | 0.7 | single stochastic draw |
| 3 | edge_enhance | original | 0.0 | image diversity, deterministic |
| 4 | grayscale | original | 0.0 | image diversity, deterministic |
| 5 | jpeg_recompress | original | 0.0 | image diversity, deterministic |
| 6 | brightness_contrast | original | 0.0 | image diversity, deterministic |
| 7 | rotation_90 | original | 0.0 | image diversity, deterministic (safe at T=0) |
| 8 | edge_enhance | hardcoded_paraphrase | 0.0 | combined diversity, deterministic |

**Key differences from Run 1:**
- All T=0.7 → T=0.0; candidate 2 (was greedy) → T=0.7 (stochastic)
- Greedy baseline is now candidate 0 (original/original/T=0)
- rotation_90 is safe at T=0 (argmax, no `torch.multinomial` call) → no OCR substitution needed
- Output written to `results/tts_hard_bench_t0/` (Run 1 data untouched)

**Models:** Qwen2.5-VL-3B and GRIT-3B (same as Run 1)
**Questions:** 30 per task (VQA, OCR, Counting) = 90 per model
**Output:** `results/tts_hard_bench_t0/{model}_results.jsonl`
**Status:** COMPLETE — see report.md § "Run 2 — T=0 Ablation"

---

Step 1 — Find the questions                                                                                                                                                                         
                                                                                                                                                                                                      
  python scripts/select_calibration_questions.py                                                                                                                                                      
                                                                                                                                                                                                      
  Two passes, one model at a time (no OOM risk):                                                                                                                                                      
                                                                                                                                                                                                      
  1. Pass 1 (Qwen 3B) — scans all available questions per task, keeps those where 1–3/5 samples are correct (20–60%). For OCR, additionally requires at least 1 sample to match answers_all.          
  2. Saves intermediate results to results/calibration/calibration_pass1.jsonl (safe to resume if it crashes).
  3. Pass 2 (Qwen 7B) — re-runs only the Pass-1 survivors, applies the same accuracy filter. Stops per task the moment 10 questions are selected.                                                     
                                                                                                                                                                                                      
  Output: results/calibration/selected_questions.jsonl                                                                                                                                                
                                                                                                                                                                                                      
  ---                                                                                                                                                                                                 
  Step 2 — Measure the entropy delta                              

  python experiments/run_study_a_entropy.py
                                                                                                                                                                                                      
  Reads the selected questions, runs Qwen 3B then GRIT (unloads between models), then prints:                                                                                                         
                                                                                                                                                                                                      
  Task         Qwen3B H     GRIT H          Δ   Δ > 0?                                                                                                                                                
  ──────────────────────────────────────────────────────                                                                                                                                              
    vqa           1.200       1.850      +0.650     YES ✓
    ocr           1.800       2.300      +0.500     YES ✓                                                                                                                                             
    counting      0.500       0.900      +0.400     YES ✓                                                                                                                                             
                                                                                                                                                                                                      
    Δ > 0 on 3/3 tasks                                                                                                                                                                                
    Go criterion (Δ > 0 on ≥ 2/3): GO                                                                                                                                                                 
                                                                                                                                                                                                      
  Output: results/study_a/entropy_results.jsonl + results/study_a/summary.json
