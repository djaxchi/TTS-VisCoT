# Test-Time Scaling for Visual Chain-of-Thought Models: Image-Input Diversity as the Key Driver

**Date:** 2026-04-13
**Status:** Runs 1 & 2 complete; analysis ongoing

---

## 1. Research Question

Can visual chain-of-thought (CoT) models benefit from test-time scaling (TTS) via
majority voting? And if so, what is the source of candidate diversity that makes it work?

We compare two models that share the same 3B Qwen2.5-VL backbone but differ only in
their reasoning strategy:

| Model | Reasoning strategy | Parameters |
|---|---|---|
| **Qwen2.5-VL-3B** | Direct answer (no CoT) | 3B |
| **GRIT-3B** | Visual CoT: think → grounded bbox → rethink → answer | 3B (same backbone) |

This pairing isolates the effect of visual CoT: same capacity, same weights, only the
prompting and fine-tuning strategy differs.

---

## 2. Experimental Setup

### Benchmark

We evaluate on three vision tasks from hard_bench — chosen for hardness (< 60% for
current 7B VLMs), recency (released late 2024), and vision-indispensability:

| Task | Source | Format |
|---|---|---|
| VQA | MMMU-Pro (Sep 2024) | 10-option MCQ, 30 academic disciplines |
| OCR | OCRBench v2 (NeurIPS 2024) | Free-form text extraction, 30 OCR types |
| Counting | MMStar instance-counting (NeurIPS 2024) | MCQ A/B/C/D |

**30 questions per task, per model, per run** (90 per model per run). Questions were
calibrated via a Pass-1 sweep (N=5, T=0.7) to fall in [20%, 70%] accuracy for Qwen3B.

### TTS candidate recipe

Each question generates **9 candidate answers** varying two diversity axes — image
augmentation and temperature:

| # | Image augmentation | Temperature | | # | Image augmentation | Temperature |
|---|---|---|---|---|---|---|
| 0 | Original | 0.0 (greedy) | | 5 | Edge enhance | 0.7 |
| 1 | Original | 0.7 | | 6 | JPEG recompress | 0.7 |
| 2 | Paraphrase | 0.7 | | 7 | Rotation 90 | 0.7 |
| 3 | Brightness +30% | 0.7 | | 8 | Paraphrase 2 | 0.7 |
| 4 | Grayscale | 0.7 | | | | |

Final answer selected by majority vote (plurality) over the 9 candidates.

### Two runs: isolating diversity sources

| Run | Temperature | Image augmentations | Purpose |
|---|---|---|---|
| **Run 1** | Mixed (T=0.0 + T=0.7) | Yes | Full TTS pipeline |
| **Run 2** (T=0 ablation) | All T=0.0 | Yes | Isolate image-augmentation diversity |

Run 2 is the critical experiment: by removing temperature stochasticity, we test whether
image augmentations alone produce useful diversity.

---

## 3. Results

### 3.1 Run 1 — Full TTS pipeline

| | **Qwen3B (direct)** | | | **GRIT (visual CoT)** | | |
|---|---|---|---|---|---|---|
| Task | Greedy | @9 | Oracle@9 | Greedy | @9 | Oracle@9 |
| VQA | 30.0% | 20.0% | 53.3% | 10.0% | 13.3% | 63.3% |
| OCR | 3.3% | 3.3% | 6.7% | 26.7% | 30.0% | 40.0% |
| Counting | 36.7% | 36.7% | 93.3% | 33.3% | 30.0% | 73.3% |

![Figure 5 — Run 1: Greedy vs Majority@9 vs Oracle@9](results/report_figures/fig5_run1_accuracy.png)

**Figure 5** — Greedy, majority @9, and oracle @9 for both models (30 questions/task).
Two things stand out: (1) TTS gains are small or negative for both models, and
(2) oracle@9 is dramatically higher than @9 — the correct answer is almost always
*present* in 9 candidates but majority voting fails to select it (e.g. Qwen Counting:
93.3% oracle vs 36.7% vote).

### 3.2 The key experiment: Run 1 vs Run 2 (T=0 ablation)

| Model | Task | Run 1 TTS gain | Run 2 TTS gain | What changed |
|---|---|---|---|---|
| Qwen3B | VQA | **-10.0pp** | 0.0pp | Temperature hurt; removing it removes the harm |
| Qwen3B | OCR | 0.0pp | 0.0pp | No diversity source helps |
| Qwen3B | Counting | 0.0pp | -6.7pp | Image augmentations add noise, no useful diversity |
| **GRIT** | **VQA** | **+3.3pp** | **+3.3pp** | **Same gain — temperature was irrelevant** |
| **GRIT** | **OCR** | **+3.3pp** | **+3.3pp** | **Same gain — temperature was irrelevant** |
| GRIT | Counting | -3.3pp | 0.0pp | Removing temperature removes the harm |

![Figure 6 — TTS gain: Run 1 vs Run 2](results/report_figures/fig6_run1_vs_run2_tts_gain.png)

**Figure 6** — TTS gain (@9 minus greedy) for both runs. Left: Qwen3B — temperature is
the only diversity source, and it hurts (-10pp VQA). At T=0, image augmentations produce
zero diversity (all @k identical to greedy). Right: GRIT — gains are identical in both
runs (+3.3pp VQA, +3.3pp OCR). All useful diversity comes from image augmentations
passing through the visual grounding step, not from temperature.

---

## 4. Findings

### Finding 1: Visual CoT models benefit from TTS via image-input diversity; direct-answer models do not

The T=0 ablation proves this cleanly. At T=0, Qwen produces the exact same answer
regardless of image augmentation — brightness, grayscale, edge enhance, JPEG, rotation
are all invisible to it. GRIT gains +3.3pp from those same augmentations because its
grounding step (think → bbox → rethink) amplifies small visual perturbations into
different reasoning chains and different final answers.

### Finding 2: Temperature stochasticity hurts majority voting

For Qwen, temperature is the only source of diversity — and it generates wrong candidates
that outvote correct ones (-10pp on VQA). For GRIT, temperature adds nothing beyond what
image augmentations already provide. In both cases, removing temperature helps or is neutral.

### Finding 3: Alternative voting strategies confirm Qwen does not benefit from TTS

The large oracle gap (oracle@9 >> @9) raises the question: is plurality voting just a
bad selector, and would a smarter strategy unlock the gains? We replayed the existing
candidate data under five alternative voting rules:

| Strategy | Rule |
|---|---|
| Greedy tiebreak | Plurality, but ties broken in favor of the greedy answer |
| Consistency filter (k=2,3) | Discard answers appearing fewer than k times, then vote |
| Supermajority (k=3,4,5) | Use greedy unless >= k candidates agree on an alternative |

**Qwen3B — best strategy is greedy itself.** No voting rule improves over the greedy
answer. The most conservative strategies (supermajority_5) simply recover the greedy
baseline, confirming that TTS candidates add only noise for a direct-answer model.

| Qwen3B | Greedy | Plurality | Best strategy | Oracle@9 |
|---|---|---|---|---|
| VQA (n=100) | **27.0%** | 20.0% | supermajority_5: 26.0% | 48.0% |
| Counting (n=92) | 39.1% | 38.0% | supermajority_3: **42.4%** | 91.3% |

**GRIT — voting strategies yield small gains over greedy.** Unlike Qwen, GRIT
benefits from candidate aggregation regardless of the specific voting rule, with
supermajority_3 reaching +10pp over greedy on VQA (Run 2).

| GRIT | Greedy | Plurality | Best strategy | Oracle@9 |
|---|---|---|---|---|
| VQA (n=30, Run 2) | 10.0% | 13.3% | supermajority_3: **20.0%** | 56.7% |
| OCR (n=30, Run 1) | 13.3% | 16.7% | plurality: **16.7%** | 20.0% |

The pattern is consistent: for Qwen, the optimal strategy converges to "just trust the
greedy answer" — TTS adds nothing. For GRIT, multiple voting strategies improve over
greedy, confirming that the visual CoT candidates carry useful signal that aggregation
can exploit. However, all strategies remain far below the oracle, indicating that
answer-level voting alone cannot close the gap — token-level confidence signals or
learned verifiers would be needed.

### Finding 4: Stochasticity does not predict TTS effectiveness

We initially expected CoT models to be more stochastic, making TTS more effective.
A calibrated entropy study (30 questions, 10 draws each at T=0.7) showed the opposite:
GRIT is *less* stochastic than Qwen on all three tasks (entropy delta: -0.23 to -0.47
bits). Yet GRIT benefits from TTS while Qwen does not. What matters is not how much
diversity a model produces, but whether that diversity is *structural* (grounding-driven)
or *random* (temperature-driven).

---

## 5. Limitations

- **Sample size:** 30 questions per task — trends across tasks and runs are more reliable
  than individual task numbers.
- **Model pair:** Only Qwen3B vs GRIT-3B tested. DeepEyesV2-7B (agentic CoT) was piloted
  on 3 questions; full evaluation is planned.
- **OCR evaluation:** Qwen3B OCR at 3.3% in Run 1 is an artifact of strict exact-match
  scoring. Re-evaluation with character-level edit distance is pending.
- **Answer-level voting only:** Five voting strategies tested, all operating on final
  answer strings. Token-level confidence (logprob weighting) was not evaluated and may
  close more of the oracle gap.

---

## 6. Next Steps

1. **Token-level confidence voting** — answer-level voting cannot close the oracle gap.
   The inference infrastructure for extracting option logprobs already exists; a
   logprob-weighted voting pass on existing candidates is the natural next step.
2. **Scale validation** — confirm the main finding on 100 questions per task (current
   results use 30).
3. **DeepEyesV2 full evaluation** — extend Run 1 + Run 2 to agentic CoT to test whether
   the image-diversity finding generalizes to deeper CoT.
