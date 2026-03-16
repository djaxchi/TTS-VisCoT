# Experiment 01 — Visual Chain-of-Thought Model Comparison

> **Status:** completed · **Date:** 2026-03-15 · **Run file:** `results/comparison/run2_judged.json`

---

## 1. Motivation

Visual Question Answering (VQA) is a challenging multimodal task that requires
integrating spatial understanding, commonsense reasoning, and language generation.
A promising direction is to equip vision-language models with explicit reasoning
traces — *visual chain-of-thought* (Visual CoT) — so that the model does not jump
directly to an answer but instead grounds its response through intermediate
perceptual steps.

In this experiment we investigate whether adding a reasoning mechanism — either a
bounding-box-guided CoT or a tool-calling agentic loop — systematically improves
accuracy over a plain direct-answering baseline, and at what computational cost.

---

## 2. Experimental Setup

### 2.1 Models

We benchmarked four models spanning three families of visual reasoning:

| Model | Type | Parameters | Quantisation |
|---|---|---|---|
| **Qwen2.5-VL (7B, no CoT)** | Direct VLM (baseline) | 7 B | 8-bit |
| **VisCoT (7B)** | Bounding-box Visual CoT | 7 B | native |
| **GRIT (3B)** | Grounded reasoning CoT | 3 B | 8-bit |
| **DeepEyesV2-RL (7B)** | Tool-calling agentic CoT | 7 B | 8-bit |

- **Qwen2.5-VL** serves as the *no-CoT* reference: it answers directly from the
  image in a single forward pass.
- **VisCoT** augments Qwen-class weights with explicit visual grounding — the model
  first predicts a bounding box region relevant to the question, then produces its
  answer conditioned on that crop.
- **GRIT** is a smaller (3 B) grounded-reasoning model that generates structured
  step-by-step rationales before committing to an answer.
- **DeepEyesV2-RL** implements a multi-turn *agentic* loop (up to 5 turns per
  question): the model can write Python code to programmatically inspect the image
  (e.g. crop, count pixels, read text regions) and incorporate the tool output
  before producing a final `<answer>` tag.  It was trained with reinforcement
  learning (GRPO) on top of a supervised fine-tuned checkpoint.

### 2.2 Tasks and Data

We evaluated on three curated tasks drawn from the VisCoT benchmark, with
**20 samples per task** (60 per model):

| Task | Description |
|---|---|
| **VQA** | Open-ended visual questions (objects, colours, relationships) |
| **Counting** | Numeric counting of objects in the scene |
| **OCR** | Reading text present in the image |

### 2.3 Evaluation Metrics

We report two complementary accuracy measures:

- **Exact-match accuracy** — a normalised string match between the model answer and
  the reference string(s).  Fast and deterministic, but penalises valid paraphrases
  (e.g. *"The helmet is blue"* vs. reference *"light blue"* is marked incorrect).
- **LLM-judge accuracy** — we re-evaluated every answer using Qwen2.5-VL-7B-Instruct
  in a text-only judging prompt.  The judge is asked to decide YES/NO on semantic
  correctness, explicitly tolerating capitalisation, punctuation, and minor wording
  differences.  This gives a more lenient and arguably more meaningful estimate of
  true correctness.

Inference time is measured as **wall-clock seconds per sample** on a single GPU.

---

## 3. Hypothesis

We expected **DeepEyesV2-RL** to be the top performer overall.  Its agentic
tool-calling loop was designed precisely for tasks that benefit from programmatic
image inspection: counting objects (iteration over detections), OCR (crop + read),
and spatially-grounded VQA (zoom into the relevant region).  The combination of
supervised fine-tuning with RL-based reward shaping (GRPO) was further expected to
improve answer precision compared to purely SFT models.

Among the CoT models without tool use, we expected **VisCoT** and **GRIT** to
outperform the direct **Qwen2.5-VL** baseline, since structured grounding should
reduce spurious answers, particularly on Counting and OCR where spatial
precision matters most.

---

## 4. Results

### 4.1 Accuracy

![Model Comparison — Accuracy & Compute Cost](../results/comparison/run2_plot.png)

*Figure 1. Left panels: exact-match (solid) and LLM-judge (hatched) accuracy per task
and overall.  Bottom panel: mean inference time per sample on a log scale.*

**Table 1 — Overall accuracy (60 samples per model)**

| Model | Exact-match | LLM-judge |
|---|---|---|
| GRIT (3B) | **73%** | 82% |
| VisCoT (7B) | 70% | **83%** |
| Qwen2.5-VL (7B, no CoT) | 63% | 75% |
| DeepEyesV2-RL (7B) | 50% | 75% |

**Table 2 — Per-task LLM-judge accuracy**

| Model | VQA | Counting | OCR |
|---|---|---|---|
| VisCoT (7B) | **85%** | **85%** | 80% |
| GRIT (3B) | **85%** | **85%** | 75% |
| Qwen2.5-VL (7B, no CoT) | 55% | **90%** | **80%** |
| DeepEyesV2-RL (7B) | 65% | 80% | **80%** |

### 4.2 Compute Time

**Table 3 — Mean inference time per sample**

| Model | VQA | Counting | OCR | Overall |
|---|---|---|---|---|
| Qwen2.5-VL (7B, no CoT) | 1.1 s | 0.5 s | 1.9 s | **1.2 s** |
| VisCoT (7B) | 2.1 s | 2.0 s | 2.0 s | 2.0 s |
| GRIT (3B) | 13.6 s | 10.1 s | 11.5 s | 11.7 s |
| DeepEyesV2-RL (7B) | 90.5 s | 83.0 s | 73.3 s | **82.2 s** |

The compute gap is striking: DeepEyesV2-RL is **~69× slower** than VisCoT and
**~68× slower** than the direct baseline, yet does not achieve higher accuracy.

---

## 5. Analysis

### 5.1 VisCoT and GRIT lead — CoT helps, but modestly

Both CoT models outperform the direct baseline on VQA, confirming that visual
grounding does help for open-ended reasoning questions.  The gap is most visible
in the LLM-judge scores (VisCoT: 85%, GRIT: 85% vs. Qwen: 55% on VQA), suggesting
the baseline tends to produce verbose answers that the exact-match metric penalises,
while CoT models produce crisper responses.

Notably, **GRIT achieves competitive results with only 3 B parameters**, hinting
that the grounding mechanism, not raw model capacity, is the primary driver of
quality.

### 5.2 DeepEyesV2-RL underperforms its expected potential

Contrary to our hypothesis, DeepEyesV2-RL is the **worst performer on exact-match
(50%)** and ties with the direct baseline on LLM-judge (75%).  We identify two
plausible explanations:

1. **Cropped-region hallucination.** The agentic loop instructs the model to
   programmatically crop sub-regions of the image for closer inspection.  If the
   initial crop does not contain the answer (e.g. the bounding box misses the
   relevant object), all subsequent reasoning steps operate on uninformative pixels,
   compounding the error rather than correcting it.  This is a *garbage-in,
   garbage-out* failure mode specific to tool-calling CoT.

2. **Answer-format mismatch.** The model outputs verbose agentic traces with
   `<code>` / `<answer>` tags.  The exact-match metric strips these tags but the
   resulting answer strings tend to be longer and less normalised, widening the
   exact-match gap relative to the LLM-judge gap (25 pp for DeepEyesV2-RL vs. ~13 pp
   for the other models).

3. **Limited turns.** We capped the agentic loop at 5 turns per question to keep
   inference tractable.  On complex VQA questions the model may need more iterations
   to converge; we observed several cases where turn 5 exhausted without a confident
   `<answer>`, resulting in an empty prediction.

### 5.3 OCR: tool-calling is no silver bullet

DeepEyesV2-RL and Qwen both reach 80% LLM-judge on OCR — tied with VisCoT and
above GRIT (75%).  This is noteworthy because OCR is exactly the kind of fine-grained
task one would expect tool-calling to excel at (crop the text region, read it).
The tie suggests the base Qwen2.5-VL backbone is already strong at OCR, and the
overhead of the agentic loop adds ~73 s/sample without a systematic benefit.

### 5.4 Exact-match vs. LLM-judge gap

Across all models, LLM-judge consistently awards 10–15 percentage points more than
exact-match.  The gap is largest for DeepEyesV2-RL (+25 pp overall), confirming
that its verbose outputs are semantically correct more often than the string match
suggests, but the model still cannot match the conciseness of VisCoT or GRIT.

---

## 6. Conclusions

- **Visual CoT helps**, particularly for VQA, and even a 3 B grounded model (GRIT)
  can match or exceed a 7 B direct baseline at a modest compute overhead (~10×).
- **VisCoT offers the best accuracy/compute trade-off**: 83% LLM-judge accuracy
  at only 2 s/sample.
- **Agentic tool-calling (DeepEyesV2-RL) did not fulfil its promise** on this
  benchmark.  The crop-based inspection strategy is a liability when the initial
  region selection is wrong, and the compute cost (~82 s/sample) is prohibitive
  for any large-scale evaluation.
- The **exact-match metric consistently under-estimates** model quality; LLM-judge
  correlation is imperfect but provides a more faithful signal, especially for
  models that generate complete sentences instead of single-word answers.
