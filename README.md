# TTS-VisCoT: Test-Time Scaling for Visual Reasoning Models

> **Research project** (IFT6765 — Polytechnique Montréal, March 2026)
> Authors: Djalil Chikhi, Youssef Laatar

Investigating whether test-time scaling (TTS) — running multiple perturbed inference passes and aggregating via majority vote — improves the accuracy of visual reasoning models on VQA, OCR, and counting tasks.

---

## Key findings

- **GRIT (3B)** is the primary TTS candidate: its explicit reasoning structure benefits most from multi-candidate aggregation.
- **Qwen2.5-VL (3B/7B)** serves as the direct-answer baseline.
- TTS with 9 candidates (7 augmentation variants + 2 paraphrase variants) yields consistent gains on VQA and counting; OCR is more sensitive to image distortions.
- Full results and plots: [`results/comparison/`](results/comparison/) and [`results/tts/`](results/tts/).

### Entropy pilot — stochasticity vs CoT depth (April 2026)

We ran a stochasticity pilot on 3 open-ended questions (VQA, OCR, counting) with 10 samples at temperature=0.7 per model, measuring both **answer entropy** (diversity of final answers) and **token-level generation entropy** (mean Shannon entropy of per-token logit distributions).

| Model | Task | Answer H (bits) | Token H (bits) | Mean gen time |
|---|---|---|---|---|
| Qwen2.5-VL-3B (no CoT) | VQA | 3.32 | 0.731 | 11.3s |
| Qwen2.5-VL-3B | OCR | 2.65 | 0.590 | 2.2s |
| Qwen2.5-VL-3B | Counting | 0.00 | 0.287 | 0.5s |
| GRIT-3B (visual CoT) | VQA | 3.32 | 0.555 | 46.0s |
| GRIT-3B | OCR | 2.16 | 0.740 | 11.2s |
| GRIT-3B | Counting | 0.97 | 0.781 | 10.8s |
| DeepEyesV2-7B (agentic CoT) | VQA | 1.57 | **0.156** | 61.9s |
| DeepEyesV2-7B | OCR | 2.85 | **0.138** | 141.6s |
| DeepEyesV2-7B | Counting | 1.36 | **0.155** | 44.9s |

**Findings:**

1. **Token entropy is inversely correlated with CoT depth** — contrary to the original hypothesis. DeepEyesV2 (~0.15 bits/token) is far more internally confident per token than Qwen (~0.6 bits) or GRIT (~0.7 bits). Its stochasticity is *structural* (different reasoning paths, 2–3 turns) rather than *distributional* (token-level noise).

2. **Deeper CoT improves accuracy** — DeepEyesV2 is the only model that correctly solves the hard VQA finance question (6.67% ≈ GT 6.66%); Qwen and GRIT both fail entirely. Its lower answer entropy reflects *being right consistently*, not reduced diversity.

3. **The TTS hypothesis needs reframing** — token-level entropy is a poor proxy for TTS-relevant stochasticity. The metric that matters is *oracle probability* (does at least one of N runs get it right?), not per-token uncertainty.

Raw data: [`results/entropy_pilot/`](results/entropy_pilot/)

---

## Models benchmarked

| Model | Size | Reasoning style |
|---|---|---|
| Qwen2.5-VL | 3B / 7B | Direct answer (baseline) |
| GRIT | 3B | Explicit CoT before answering |
| VisCoT | 7B | Bounding-box grounded reasoning |
| DeepEyesV2-RL | 7B | Agentic tool-calling loop |

---

## Datasets

### hard_bench — primary benchmark

Three datasets were selected for their hardness (7B VLMs score 35–60%), recency (published at or after model training cutoffs), and vision-indispensability (image required to answer).

| File | Task | Dataset | Size | Why |
|---|---|---|---|---|
| `data/hard_bench/vqa_100.jsonl` | VQA | MMMU-Pro (standard, 10-option) | 100 | 30 academic disciplines; Qwen2.5-VL-7B ~38–41% |
| `data/hard_bench/ocr_100.jsonl` | OCR | OCRBench v2 | 100 | 30 OCR types (handwriting, artistic text, irregular layouts); NeurIPS 2024 |
| `data/hard_bench/counting_100.jsonl` | Counting | MMStar — instance counting | **92** | Full `l2_category="instance counting"` subset; vision-indispensable by construction; NeurIPS 2024 |

Samples are drawn round-robin across subjects/types for diversity (seed 42).
Images are not committed to git — they are fetched from HuggingFace on first access and cached under `data/hard_bench/images/`.

### TreeBench — harder VQA evaluation set

Used for out-of-distribution evaluation. Metadata lives in `data/treebench_samples/`; images are gitignored.

---

## TTS approach

For each input, generate **9 candidates** using a fixed recipe of input perturbations, then aggregate via **majority vote**:

### Image augmentations
| ID | Transform |
|---|---|
| `edge` | Edge enhancement |
| `gray` | Grayscale conversion |
| `jpeg` | JPEG recompression (blur) |
| `brightness` | Brightness + contrast shift |
| `rotate90` | 90° rotation |

### Text (prompt) variants
| ID | Method |
|---|---|
| `hardcoded_paraphrase` | Pre-written rephrase of the question |
| `model_paraphrase` | LLM-generated rephrase (cached) |

---

## Repository layout

```
TTS-VisCoT/
├── configs/
│   ├── datasets/           treebench.yaml
│   ├── models/             grit.yaml, viscot.yaml, deepeyes_v2.yaml
│   └── experiments/        baseline.yaml, tts.yaml, comparison.yaml
│
├── data/
│   ├── hard_bench/         vqa_100.jsonl, ocr_100.jsonl, counting_100.jsonl
│   └── treebench_samples/  metadata.jsonl (images gitignored)
│
├── experiments/
│   ├── run_model_benchmark.py      Baseline comparison across all models
│   ├── run_tts_eval.py             TTS evaluation on VGQAV2
│   ├── run_test_time_scaling.py    Full TTS scaling sweep
│   ├── run_tts_hard.py             TTS on hard subsets
│   └── run_tts_treebench.py        TTS on TreeBench
│
├── results/
│   ├── comparison/         ModelBenchmark.json + final figures (figA–figG)
│   └── tts/                TTS.json, TTS_Hard.json + scaling plots
│
├── scripts/
│   ├── plot_results.py             Generate comparison figures
│   ├── plot_tts_scaling.py         TTS scaling curves
│   ├── plot_tts_hard_candidates.py Hard-subset analysis
│   ├── plot_presentation.py        Slide-ready figures
│   ├── build_static_paraphrase_cache.py  Pre-compute question paraphrases
│   └── export_treebench_questions.py     Export TreeBench samples
│
├── src/
│   ├── augment_image.py    Image perturbation specs + generators
│   ├── augment_text.py     Prompt paraphrase generators
│   ├── pipeline_tts.py     Core TTS pipeline (build_candidate_inputs, run_tts_pipeline)
│   ├── voting_tts.py       Voting utilities (VoteStats, compute_vote_stats)
│   ├── utils_normalize.py  Answer normalization (open-ended + MCQ)
│   ├── token_aggregation.py  Token-level logit aggregation (experimental)
│   ├── check_token_support.py  Check if model exposes token probabilities
│   ├── data/
│   │   ├── datasets/       base.py, viscot_benchmark.py, treebench.py, treebench_export.py
│   │   └── augmentation/   base.py, image_aug.py, text_aug.py, views.py
│   ├── eval/
│   │   ├── metrics.py      AccuracyMetrics, BBoxMetrics, RobustnessMetrics
│   │   ├── tts_eval.py     make_predict_fn, evaluate_one, compute_summary
│   │   ├── voting_replay.py  Replay saved candidates under different voting strategies
│   │   ├── token_trace.py  Token-level agreement analytics (experimental)
│   │   ├── tts_trace_metrics.py  Candidate trace analytics
│   │   └── vqa_eval.py     VQA string-match evaluation
│   ├── methods/
│   │   ├── baseline.py     Single-pass inference
│   │   └── tts/            sampling.py, scaling.py, open_ended.py
│   ├── models/
│   │   ├── base.py         BaseVisualCoTModel
│   │   ├── direct_vlm.py   Qwen2.5-VL wrapper
│   │   ├── grit.py         GRIT wrapper
│   │   ├── viscot.py       VisCoT wrapper
│   │   └── deepeyes_v2.py  DeepEyesV2 agentic wrapper
│   ├── voting/
│   │   ├── majority.py, bbox_consensus.py, normalize.py
│   └── utils/
│       ├── io.py, logging.py
│
└── tests/
    ├── test_run_comparison.py      Benchmark checkpoint/resume logic
    ├── test_run_tts_eval.py        Paraphrase cache + candidate view saving
    ├── test_tts_eval.py            make_predict_fn, evaluate_one, compute_summary
    ├── test_tts_pipeline.py        build_candidate_inputs, run_tts_pipeline, voting
    ├── test_voting_replay.py       Voting replay + reliability weights
    ├── test_treebench_export.py    TreeBench export utility
    ├── test_token_aggregation.py   Token-level aggregation (experimental)
    ├── test_token_trace.py         Token trace analytics
    └── test_tts_trace_metrics.py   Candidate trace metrics
```

---

## Running experiments

### 1. Model benchmark (baseline comparison)

```bash
python experiments/run_model_benchmark.py \
    --n 100 \
    --save-output results/comparison/ModelBenchmark.json
```

### 2. TTS evaluation on VGQAV2

```bash
python experiments/run_tts_eval.py \
    --data-dir data/VGQAV2 \
    --benchmark-task vqa \
    --model-type grit \
    --save-dir results/tts_eval/grit_vqa
```

### 3. Full TTS scaling sweep

```bash
python experiments/run_test_time_scaling.py \
    --save-output results/tts/TTS.json
```

### 4. Plot final figures

```bash
python scripts/plot_results.py
python scripts/plot_tts_scaling.py
```

### 5. Run all tests

```bash
pytest tests/ -v
```

---

## Hardware requirements

- GRIT / VisCoT / Qwen2.5-VL (3B): ≥ 8 GB VRAM
- Qwen2.5-VL (7B) / VisCoT (7B): ≥ 16 GB VRAM
- DeepEyesV2-RL (7B): ≥ 16 GB VRAM (`load_in_8bit=True` default)

---

## License

MIT
