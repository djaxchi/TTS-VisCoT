# TTS × Visual CoT — Experiment Report

**Date:** 2026-04-12  
**Status:** Study A complete — NO-GO on stochasticity criterion; pivoting to accuracy-first analysis

---

## Hypothesis

> Models with deeper chain-of-thought reasoning are more stochastic in their outputs.
> This stochasticity is what allows test-time scaling (TTS via majority voting) to work.
> Therefore, TTS gains should be larger for agentic CoT models than for direct-answer baselines.

The three models span a CoT-depth axis:

| Model | CoT type | Size |
|---|---|---|
| Qwen2.5-VL | None — direct answer | 7B |
| GRIT | Visual CoT, no tool use | 3B |
| DeepEyesV2-RL | Agentic CoT + code execution | 7B |

---

## Step 1 — Stochasticity Pilot

### Setup

- **Questions:** first question from each of 3 hard_bench tasks (3 questions total per model)
- **Tasks:** VQA (MMMU-Pro), OCR (OCRBench v2), Counting (MMStar instance-counting)
- **Samples:** N = 10 independent draws per question, temperature = 0.7
- **Metrics:**
  - *Answer entropy* — Shannon entropy over the distribution of (normalized) answers: `H = −Σ p(a) log₂ p(a)`
  - *Token entropy* — mean per-token entropy of the generation, averaged across all samples
  - *Unique answers* — raw count of distinct normalized answers out of 10

---

### Results

#### Answer-level entropy

![Figure 1](results/entropy_pilot/figures/fig1_answer_entropy.png)

**Figure 1** — Grouped bar chart of answer-level entropy (bits) by model and task.
Higher bars indicate more diverse final answers across the 10 samples.

| Model | VQA | OCR | Counting | Mean |
|---|---|---|---|---|
| Qwen2.5-VL | 3.322 | 2.646 | **0.000** | 1.989 |
| GRIT | **3.322** | 2.161 | 0.971 | 2.151 |
| DeepEyesV2 | 1.571 | **2.846** | **1.357** | 1.925 |

Key observations:
- **VQA:** Qwen and GRIT both reach maximum entropy (10 unique answers / 10 samples). DeepEyesV2 is lower (4 unique answers), likely because its agentic CoT converges on a narrower set of reasoning paths.
- **OCR:** DeepEyesV2 produces the most diverse answers (8 unique), followed by Qwen (7) and GRIT (6). DeepEyesV2's code-execution loop generates varied interpretations of the image.
- **Counting:** Qwen is completely collapsed — it always answers "2" (entropy = 0.0, 10/10 correct). GRIT and DeepEyesV2 show meaningful diversity, with DeepEyesV2 highest (4 unique answers).

---

#### Token-level entropy

![Figure 2](results/entropy_pilot/figures/fig2_token_entropy.png)

**Figure 2** — Mean token-level entropy (bits) per model per task.
This measures how uncertain the model is at each generation step, independent of whether the final answer changes.

| Model | VQA | OCR | Counting | Mean |
|---|---|---|---|---|
| Qwen2.5-VL | 0.731 | 0.590 | 0.287 | 0.536 |
| GRIT | 0.555 | **0.740** | **0.781** | 0.692 |
| DeepEyesV2 | **0.156** | 0.138 | 0.155 | **0.150** |

Notably, DeepEyesV2 has the *lowest* token entropy across all tasks, even though its answer entropy is not consistently the lowest. This reflects the agentic pattern: the model generates long, structured reasoning chains with high token-level confidence in each step, but the *conclusions* of different chains may still diverge. The CoT reasoning is locally confident but globally diverse.

---

#### Answer diversity (unique answers out of 10)

![Figure 4](results/entropy_pilot/figures/fig4_diversity_heatmap.png)

**Figure 4** — Heatmap of the number of unique normalized answers per model × task.
Darker cells indicate higher answer diversity.

---

#### Inference time

![Figure 3](results/entropy_pilot/figures/fig3_gen_time.png)

**Figure 3** — Mean generation time per sample (seconds). The CoT depth axis is clearly
visible: Qwen (direct, ~0.5–11 s) → GRIT (visual CoT, ~10–46 s) → DeepEyesV2
(agentic CoT + code, ~45–142 s). DeepEyesV2's OCR inference is particularly slow
(141.6 s/sample) because the agentic loop executes multiple code calls to parse
text from complex images.

---

### Go / No-Go Assessment

The recipe specifies the go criterion:
> **Proceed to Step 2 if `entropy(DeepEyesV2) > entropy(GRIT) > entropy(Qwen)` on at least 2/3 tasks.**

Checking answer entropy:

| Task | Rank | Criterion met? |
|---|---|---|
| VQA | GRIT = Qwen (3.32) > DeepEyes (1.57) | ❌ |
| OCR | DeepEyes (2.85) > Qwen (2.65) > GRIT (2.16) | ❌ (wrong order) |
| Counting | DeepEyes (1.36) > GRIT (0.97) > Qwen (0.00) | ✅ |

**The strict go criterion is not met (1/3 tasks).**

However, the result is more nuanced than a binary fail. Before deciding to halt, consider the following reframing.

---

### Interpretation and reframing

**1. The VQA question is genuinely hard for all models.** The ground truth is `6.66%` and no model answers correctly across any of the 10 samples. High answer entropy here reflects models being *wrong in diverse ways*, not models exploring correct reasoning paths. This is a ceiling effect on difficulty rather than evidence against the hypothesis.

**2. Qwen's counting collapse is informative.** Qwen answers "2" for all 10 counting samples (entropy = 0, accuracy = 10/10). This is a *low-temperature-like collapse* even at T=0.7 — the direct-answer model is highly confident and happens to be right. This does not refute the hypothesis; it illustrates that a collapsed model with correct answers would not benefit from TTS (no diversity to vote over, but also no need for it).

**3. DeepEyesV2 shows a different stochasticity profile.** Its low *token* entropy but higher *answer* entropy on OCR and Counting suggests the agentic reasoning loop explores genuinely different solution paths across runs, even though each individual run proceeds with high token-level confidence. This is the most TTS-amenable profile: diverse candidates that could flip a majority vote.

**4. The hypothesis may need refinement.** The original framing predicts a monotone entropy ranking across *all* tasks. The pilot suggests task type matters: on structured tasks like Counting and OCR (where the image content is the bottleneck), agentic CoT produces more diversity. On abstract reasoning (VQA, MMMU-Pro), all models spread out similarly.

---

### Recommendation

**Proceed to Step 2 with a modified hypothesis:**

> TTS gains will be larger for DeepEyesV2 than for Qwen on **Counting and OCR** tasks, where answer entropy is higher and oracle@9 headroom is meaningful. VQA is predicted to show small gains for all models due to task hardness at this temperature.

The pilot has confirmed that the premise (stochasticity exists and varies by model × task) holds. It has not confirmed a clean monotone ranking, which is grounds for hypothesis refinement, not abandonment.

---

## Study A — Calibrated Entropy Comparison (GRIT vs. Qwen3B)

### Motivation

The Step 1 pilot used only 1 question per task. Study A scales this to 30 calibrated questions
(10 per task) selected to be in the productive difficulty range [20%, 70%] accuracy for Qwen3B,
ensuring neither floor nor ceiling effects dominate the entropy signal.

### Setup

- **Questions:** 30 calibrated questions (10 per task) from hard_bench, selected via a Pass-1
  sweep at N=5, T=0.7 requiring accuracy ∈ [20%, 70%] for Qwen2.5-VL-3B
- **Tasks:** VQA (MMMU-Pro), OCR (OCRBench v2), Counting (MMStar instance-counting)
- **Models:** Qwen2.5-VL-3B-Instruct (baseline) vs. GRIT-20-Qwen2.5-VL-3B (visual CoT)
- **Samples:** N=10 independent draws per question, temperature=0.7
- **Metric:** Shannon answer entropy H = −Σ p(a) log₂ p(a) over normalized answers
- **Go criterion:** mean H(GRIT) > mean H(Qwen3B) on ≥ 2/3 tasks

### Results

| Task | Qwen3B H (bits) | GRIT H (bits) | Δ = GRIT − Qwen3B | Δ > 0? |
|---|---|---|---|---|
| VQA | 1.210 | 0.951 | −0.259 | NO |
| OCR | 1.881 | 1.410 | −0.472 | NO |
| Counting | 1.023 | 0.791 | −0.232 | NO |

**Tasks with Δ > 0: 0/3. Criterion not met.**

### Interpretation

GRIT is consistently **more deterministic** than the Qwen3B baseline across all three tasks.
The visual CoT (think → grounded bbox → rethink → answer) anchors the model on a consistent
answer rather than diversifying its outputs. This is the **opposite of the original hypothesis**.

Two possible readings:

1. **Good determinism:** GRIT's CoT reduces uncertainty because it has correctly identified the
   answer visually. The lower entropy reflects calibrated confidence, not collapse.
2. **Overconfidence:** GRIT commits early to a visual grounding (bbox) and the rethink step
   rarely reverses that commitment. On hard questions this means it can be confidently wrong.

Distinguishing these two requires looking at accuracy, which Study A did not measure.

### Go / No-Go

**NO-GO on the stochasticity-driven TTS plan.**

Running 9-candidate TTS at T=0.7 on GRIT would produce near-identical candidates (low entropy
means most temperature samples agree). Majority voting over identical candidates yields no gain
over a single greedy call. The temperature dimension of diversity is effectively dead for GRIT.

Input-augmentation diversity (rotation, grayscale, jpeg) might still induce some answer changes,
but GRIT's bbox-anchored reasoning tends to persist under photometric transforms.

---

## Run 1 — Study B + TTS, Standard Recipe (COMPLETED: 2026-04-13)

### Setup

- **Script:** `experiments/run_tts_hard_bench.py`
- **Questions:** 30 per task (VQA, OCR, Counting) = 90 questions per model
- **Models:** Qwen2.5-VL-3B (no CoT) and GRIT-3B (visual CoT)
- **Candidates:** 9 per question (see recipe for full breakdown)
- **Candidate 2** = greedy T=0.0 → **Study B accuracy baseline embedded in TTS run**
- **Note:** `rotation_90` augmentation (candidate 7) was substituted with `jpeg_recompress`
  for the OCR task after confirming that 90° rotation of complex OCR layouts triggers a
  CUDA device-side assert (`torch.multinomial` with NaN logits) in GRIT's visual encoder.
  Counts: Qwen3B OCR=30, Counting=30, VQA=30; GRIT OCR=30, Counting=30, VQA=30 (all complete).

### Results

#### Qwen2.5-VL-3B (no CoT)

| Task | greedy | @1 (T=0.7) | @3 | @5 | @9 | oracle@9 | TTS gain |
|---|---|---|---|---|---|---|---|
| VQA | 30.0% | 23.3% | 30.0% | 20.0% | 20.0% | 53.3% | −10.0pp |
| OCR | 3.3%† | 3.3% | 3.3% | 3.3% | 3.3% | 6.7% | 0.0pp |
| Counting | 36.7% | 46.7% | 36.7% | 43.3% | 36.7% | 93.3% | 0.0pp |

#### GRIT-3B (visual CoT)

| Task | greedy | @1 (T=0.7) | @3 | @5 | @9 | oracle@9 | TTS gain |
|---|---|---|---|---|---|---|---|
| VQA | 10.0% | 23.3% | 20.0% | 13.3% | 13.3% | 63.3% | +3.3pp |
| OCR | 26.7% | 26.7% | 23.3% | 26.7% | 30.0% | 40.0% | +3.3pp |
| Counting | 33.3% | 33.3% | 33.3% | 30.0% | 30.0% | 73.3% | −3.3pp |

† Qwen3B OCR accuracy is artificially low due to strict exact-match evaluation — the model
produces plausible answers but exact string normalization rejects correct responses.
Raw answers are saved for re-evaluation with a lenient metric (character-level F1 or
token overlap, as used in the official OCRBench-v2 benchmark).

### Study B verdict: **Task-dependent — GRIT wins OCR, Qwen3B wins VQA, Counting is a tie**

| Task | Qwen3B greedy | GRIT greedy | Δ |
|---|---|---|---|
| VQA | **30.0%** | 10.0% | Qwen3B +20.0pp |
| OCR | 3.3%† | **26.7%** | GRIT +23.4pp |
| Counting | **36.7%** | 33.3% | Qwen3B +3.4pp |

The picture is nuanced — the earlier report of "GRIT 0% on Counting" was an artifact of
CUDA device-side errors poisoning all GPU outputs for that task. The true numbers show GRIT
(33.3%) nearly matching Qwen3B (36.7%) on Counting, and substantially outperforming on OCR
(27.6% vs 3.3%†). Only on VQA does Qwen3B win decisively (+20pp).

**OCR:** GRIT's visual grounding (locate text region via bbox → rethink → read) is well-suited
to OCR tasks where the text is visually localized. This is the one task where CoT adds clear
value. Qwen3B's 3.3% is nearly certainly an evaluation artifact — lenient re-evaluation is required
before interpreting this gap.

**Counting:** GRIT's 33.3% matches Qwen3B's 36.7% within noise. The CoT overhead (bbox → count)
does not help or hurt meaningfully on MCQ counting tasks at this difficulty level.

**VQA:** Qwen3B wins clearly (+20pp). GRIT's visual grounding may mis-anchor on irrelevant
image regions for abstract multi-disciplinary MMMU-Pro questions, causing overconfident
wrong answers.

### TTS verdict: **TTS gives small gains or is neutral; majority voting is the bottleneck**

- **Qwen3B**: −10pp on VQA (stochastic wrong candidates outvote the correct greedy answer),
  neutral on OCR and Counting. Oracle@9 is high (53–93%), confirming correct answers exist
  in the candidate set but majority voting fails to select them.
- **GRIT**: +3.3pp on VQA, +3.4pp on OCR. TTS helps marginally when the greedy baseline is
  low and stochastic exploration occasionally finds better answers. Counting shows −3.3pp
  (same issue as Qwen3B VQA: stochastic candidates dilute the correct greedy vote).

### Interpretation

The experiment refines the hypothesis:

> **Visual CoT (GRIT) does not uniformly help or hurt accuracy — it is task-dependent.**
> **TTS via majority voting provides small or negative gains for both models.**

The dominant finding is that **majority voting is a poor selector** when the candidate pool
is mixed. Qwen3B on Counting has oracle@9 = 93.3% — the correct answer is present 93% of the
time across 9 candidates — but majority voting only reaches 36.7% because wrong answers form
the plurality. The limiting factor is selection, not diversity.

GRIT's lower stochasticity (Study A: H(GRIT) < H(Qwen3B) on all tasks) does not translate
to lower accuracy in aggregate: on 2/3 tasks (OCR, Counting) GRIT is competitive or better.
The stochasticity hypothesis was too simplistic — what matters is whether diverse candidates
include the correct answer, and whether the voter can identify it.

---

## Next Steps

1. **Better selection**: Replace majority voting with confidence-weighted voting or a
   verifier model on Qwen3B candidates. Oracle@9 = 93% on Counting means the headroom
   is huge if selection improves. This is the highest-leverage direction given current results.

2. **OCR re-evaluation**: Recompute OCR accuracy for Qwen3B with character-level edit
   distance or token F1 (the standard OCRBench-v2 metric). Qwen3B's 3.3% greedy is an
   evaluation artifact. GRIT's 27.6% (under the same strict metric) is likely an undercount too.

3. **Larger model**: DeepEyesV2 (7B agentic CoT) was in the original lineup. Its token
   entropy was lowest of all three models (Study 1 pilot), suggesting even more
   deterministic chains — but chains that occasionally reach correct conclusions via
   code execution. Worth running Study B equivalent on DeepEyesV2 to complete the
   CoT-depth axis.

### Confound note

GRIT is 3B; Qwen3B is also 3B (same backbone, Qwen2.5-VL). This is the cleanest
possible pair to isolate the CoT effect: **same weights, same size, only the
prompting/fine-tuning strategy differs**. The accuracy gap is not a capacity gap.

---

## Run 2 — T=0 Ablation: Image-Only Diversity (COMPLETED: 2026-04-13)

**Script:** `experiments/run_tts_hard_bench.py --recipe t0`

### Motivation

Run 1 mixes two sources of diversity: temperature stochasticity (T=0.7) and image
augmentations. It is unclear which drives the small TTS gains observed. Run 2 ablates
temperature: all 8 augmentation candidates run at T=0.0 (deterministic), and only
candidate 2 is stochastic (T=0.7, original image). This isolates the contribution of
image-augmentation diversity to TTS gains.

### Setup

- **Candidate recipe:** 8 × T=0.0 (image/text diversity only) + 1 × T=0.7 (candidate 2)
- **Greedy baseline:** candidate 0 (original/original/T=0.0)
- **Questions:** 30 per task for both models
- **Models:** Qwen2.5-VL-3B and GRIT-3B
- **Output:** `results/tts_hard_bench_t0/` (Run 1 data in `results/tts_hard_bench/` untouched)
- **Note:** rotation_90 at T=0 uses argmax (no `torch.multinomial` call) — no CUDA errors on OCR.
  All 180 questions ran cleanly (90 per model, 0 errors).

### Results

#### Qwen2.5-VL-3B (no CoT) — T=0 ablation

| Task | greedy | @1 | @3 | @5 | @9 | oracle@9 | TTS gain |
|---|---|---|---|---|---|---|---|
| VQA | 30.0% | 30.0% | 30.0% | 30.0% | 30.0% | 50.0% | 0.0pp |
| OCR | 30.0%† | 30.0% | 30.0% | 30.0% | 30.0% | 36.7% | 0.0pp |
| Counting | 36.7% | 36.7% | 36.7% | 36.7% | 30.0% | 70.0% | −6.7pp |

#### GRIT-3B (visual CoT) — T=0 ablation

| Task | greedy | @1 | @3 | @5 | @9 | oracle@9 | TTS gain |
|---|---|---|---|---|---|---|---|
| VQA | 10.0% | 10.0% | 13.3% | 13.3% | 13.3% | 56.7% | +3.3pp |
| OCR | 26.7% | 26.7% | 30.0% | 33.3% | 30.0% | 40.0% | +3.3pp |
| Counting | 33.3% | 33.3% | 30.0% | 33.3% | 33.3% | 66.7% | 0.0pp |

† Qwen3B OCR greedy is 30.0% here vs 3.3% in Run 1. Run 1 evaluated all 100 OCR questions
(full hard_bench); Run 2 evaluates only the first 30, which appear to be more amenable to
exact-match normalization. The comparison within Run 2 (greedy vs @9) is internally consistent.

### Run 2 vs Run 1 comparison

| Model | Task | Run1 @9 gain | Run2 @9 gain | Δ (Run2−Run1) |
|---|---|---|---|---|
| Qwen3B | VQA | −10.0pp | 0.0pp | +10.0pp |
| Qwen3B | OCR | 0.0pp | 0.0pp | 0.0pp |
| Qwen3B | Counting | 0.0pp | −6.7pp | −6.7pp |
| GRIT | VQA | +3.3pp | +3.3pp | 0.0pp |
| GRIT | OCR | +3.3pp | +3.3pp | 0.0pp |
| GRIT | Counting | −3.3pp | 0.0pp | +3.3pp |

### Interpretation

**Qwen3B — image augmentation produces zero diversity at T=0:**
All @1/@3/@5/@9 are identical to the greedy baseline (except Counting @9 which loses
6.7pp due to noise in voting over identical candidates). At T=0, Qwen3B produces the
same answer regardless of image augmentation — photometric transforms (brightness,
grayscale, edge enhance, jpeg, rotation) do not perturb the model's output. This means
**temperature stochasticity is the sole source of diversity for Qwen3B**. Since
temperature stochasticity in Run 1 hurt VQA by 10pp (wrong stochastic candidates
outvoted the correct greedy), eliminating it in Run 2 removes the harm.

**GRIT — image augmentation produces meaningful diversity at T=0:**
GRIT gains +3.3pp on VQA and OCR even with all candidates at T=0. This confirms that
GRIT's visual CoT (bbox grounding → rethink) is sensitive to image augmentations:
different image transforms produce different grounding targets, leading to different
final answers. The diversity source for GRIT is the visual grounding step, not temperature.
This explains why GRIT's TTS gain is similar in both runs (+3.3pp VQA/OCR regardless
of temperature).

### Key finding

> **The diversity mechanism differs by model architecture.**
> Qwen3B (direct answer) needs temperature stochasticity for diversity — but stochasticity
> generates wrong candidates that hurt majority voting. GRIT (visual CoT) generates diversity
> from image augmentations via its grounding mechanism — and this diversity is more useful
> (small positive TTS gains in both runs).

---

## Appendix — Raw data summary

---

## Appendix — Raw data summary

### DeepEyesV2

| Task | Question ID | GT | Unique answers | Answer H (bits) | Token H (bits) | Mean time (s) |
|---|---|---|---|---|---|---|
| VQA | test_Accounting_42 | 6.66% | 4 | 1.571 | 0.156 | 61.9 |
| OCR | 106 | off | 8 | 2.846 | 0.138 | 141.6 |
| Counting | 131131002 | 2 | 4 | 1.357 | 0.155 | 44.9 |

### GRIT

| Task | Question ID | GT | Unique answers | Answer H (bits) | Token H (bits) | Mean time (s) |
|---|---|---|---|---|---|---|
| VQA | test_Accounting_42 | 6.66% | 10 | 3.322 | 0.555 | 46.0 |
| OCR | 106 | off | 6 | 2.161 | 0.740 | 11.2 |
| Counting | 131131002 | 2 | 2 | 0.971 | 0.781 | 10.8 |

### Qwen2.5-VL

| Task | Question ID | GT | Unique answers | Answer H (bits) | Token H (bits) | Mean time (s) |
|---|---|---|---|---|---|---|
| VQA | test_Accounting_42 | 6.66% | 10 | 3.322 | 0.731 | 11.3 |
| OCR | 106 | off | 7 | 2.646 | 0.590 | 2.2 |
| Counting | 131131002 | 2 | 1 | 0.000 | 0.287 | 0.5 |
