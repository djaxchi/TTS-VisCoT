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

| Dataset | Tasks | Notes |
|---|---|---|
| VGQAV2 | VQA, OCR, Counting | 100 samples per task — primary benchmark |
| TreeBench | VQA (hard) | Harder evaluation set |

Data lives under `data/`. Image directories are gitignored (large); JSONL metadata is committed.

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
│   ├── VGQAV2/             counting_100.jsonl, ocr_100.jsonl, vqa_100.jsonl
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
