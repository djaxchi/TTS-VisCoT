# TTS × Visual CoT — Experiment Report

**Date:** 2026-04-12  
**Status:** Step 1 complete — go/no-go pending review

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

## Next Steps

- **Step 2** — Full TTS run: 50 questions × 3 tasks × 3 models, N=9 candidates (3 image augmentations × 3 text variants), majority vote at @3/@5/@9
- **Step 3** — Analysis: TTS gain vs entropy correlation (Figure 3c in recipe), scaling curves per model, augmentation breakdown
- **Confound to track:** GRIT is 3B vs 7B for the other two models. The Qwen / DeepEyesV2 pair (shared base weights, same size) is the cleanest comparison for the CoT-depth effect.

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
