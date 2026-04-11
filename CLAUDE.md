# CLAUDE.md — Guidelines for AI-assisted development in TTS-VisCoT

This file instructs any AI coding agent (Claude, Copilot, etc.) on how to
contribute to this research codebase.  Read it in full before writing any code.

---

## Core principle: Test-Driven Development (TDD)

**Every piece of logic in this repository must be written test-first.**

The workflow is strict Red → Green → Refactor:

1. **Red** — Write a failing test that precisely specifies the behaviour you need.
2. **Green** — Write the *minimum* production code that makes the test pass.
3. **Refactor** — Clean up code and tests while keeping all tests green.

Do not skip step 1.  Do not write implementation code that has no corresponding test.

---

## Repository layout

```
TTS-VisCoT/
├── configs/
│   ├── datasets/        treebench.yaml
│   ├── models/          grit.yaml, viscot.yaml, deepeyes_v2.yaml
│   └── experiments/     baseline.yaml, tts.yaml, comparison.yaml
│
├── data/
│   ├── hard_bench/                      vqa_100.jsonl, ocr_100.jsonl, counting_100.jsonl
│   ├── treebench_samples/               metadata.jsonl (images gitignored)
│   ├── treebench_paraphrase_template.jsonl
│   └── treebench_paraphrased.jsonl
│
├── experiments/
│   ├── run_model_benchmark.py    Baseline comparison across all models
│   ├── run_test_time_scaling.py  Full TTS scaling sweep (primary experiment)
│   ├── run_tts_hard.py           TTS on hard question subsets
│   └── run_tts_treebench.py      TTS on TreeBench
│
├── results/
│   ├── comparison/       ModelBenchmark.json, ModelBenchmark_judged.json + figures
│   └── tts/              TTS.json, TTS_Hard.json, TTS_TreeBench.json + scaling plots
│
├── scripts/
│   ├── prepare_hard_bench.py              Generate data/hard_bench/ JSONL files from HF
│   ├── build_static_paraphrase_cache.py   Pre-compute hardcoded paraphrase cache
│   ├── export_treebench_questions.py       Export TreeBench questions for paraphrasing
│   ├── run_ablations.sh                    Ablation sweep driver
│   └── setup_env.sh                        Environment setup
│
├── src/
│   ├── augmentation/
│   │   ├── image.py              Image perturbation specs + generators
│   │   └── text.py               Prompt paraphrase generators
│   ├── pipeline_tts.py           Core TTS pipeline
│   ├── voting_tts.py             Voting utilities (VoteStats, compute_vote_stats)
│   ├── utils_normalize.py        Answer normalization
│   ├── data/
│   │   └── datasets/             base.py, viscot_benchmark.py, treebench.py, treebench_export.py
│   ├── eval/
│   │   ├── metrics.py            AccuracyMetrics, BBoxMetrics, RobustnessMetrics
│   │   ├── tts_eval.py           make_predict_fn, evaluate_one, compute_summary
│   │   ├── tts_vote_analysis.py  3-vs-5 vote accuracy analysis
│   │   ├── voting_replay.py      Replay candidates under different voting strategies
│   │   └── vqa_eval.py           VQA string-match evaluation
│   ├── models/
│   │   ├── base.py               BaseVisualCoTModel
│   │   ├── direct_vlm.py         Qwen2.5-VL wrapper
│   │   ├── grit.py               GRIT wrapper
│   │   ├── viscot.py             VisCoT wrapper
│   │   └── deepeyes_v2.py        DeepEyesV2 agentic wrapper
│   └── utils/                    io.py, logging.py
│
└── tests/
    ├── test_run_comparison.py      Benchmark checkpoint/resume logic
    ├── test_tts_eval.py            make_predict_fn, evaluate_one, compute_summary
    ├── test_tts_pipeline.py        build_candidate_inputs, voting utilities
    ├── test_tts_vote_analysis.py   3-vs-5 vote accuracy analysis
    ├── test_voting_replay.py       Voting replay + reliability weights
    └── test_treebench_export.py    TreeBench export utility
```

---

## Benchmark datasets

### Why we replaced VGQAV2

The original benchmark (VGQAV2 — GQA/TextVQA/VQAv2 subsets) was retired because
state-of-the-art 7B VLMs score above 80 % on all three tasks, leaving no room to
measure TTS gains.  The new `hard_bench` datasets target regimes where current
models still struggle (roughly 35–60 %).

### hard_bench layout

```
data/hard_bench/
├── vqa_100.jsonl        MMMU-Pro  (100 samples, 30 academic subjects)
├── ocr_100.jsonl        OCRBench  (100 samples, 10 recognition types)
├── counting_100.jsonl   ChartQA   (100 counting-style questions)
└── images/              Local image cache — gitignored, populated on first run
```

Each JSONL row has fields: `question_id`, `question`, `answer`, `image_id`, `image_source`.
Images are **not** stored in git.  The loader (`src/data/datasets/viscot_benchmark.py`)
fetches them from HuggingFace on first access and caches them under `data/hard_bench/images/`.

To regenerate the JSONL files (e.g. after changing the sample selection):

```bash
python scripts/prepare_hard_bench.py          # all three tasks
python scripts/prepare_hard_bench.py vqa      # single task
```

### Dataset details

| Task | Dataset | HF repo | Why it's hard |
|---|---|---|---|
| VQA | **MMMU-Pro** | `MMMU/MMMU_Pro` | 10-option MCQ across 30 academic disciplines; questions that require genuine visual grounding, not text shortcuts; Qwen2.5-VL-7B ~38-41 % |
| OCR | **OCRBench v1** | `echo840/OCRBench` | 10 recognition types including handwriting, artistic text, irregular layouts, and handwritten math; Qwen2.5-VL-7B ~42/100 |
| Counting | **ChartQA (counting subset)** | `lmms-lab/ChartQA` | Counting bars/segments/data-points in real scientific charts; requires reading chart structure, not subitizing; filtered to non-trivial counts (answer > 3) |

### Image sources and loader

`src/data/datasets/viscot_benchmark.py → load_task(task, n)` dispatches on
`image_source` per row:

| `image_source` value | HF dataset fetched |
|---|---|
| `mmmu_pro` | `MMMU/MMMU_Pro`, standard (10 options), test split — matched by question `id` |
| `ocrbench` | `echo840/OCRBench`, test split — matched by sequential dataset index |
| `chartqa` | `lmms-lab/ChartQA`, test split — matched by sequential dataset index |
| `gqa` | `lmms-lab/GQA`, val_balanced_images — retained for TreeBench compatibility |

Images are saved as JPEG to `data/hard_bench/images/<source>/<image_id>.jpg` after
the first fetch; subsequent runs are fully offline.

---

## TDD rules for this project

### 1. Tests live in `tests/`

| Source file | Test file |
|---|---|
| `src/pipeline_tts.py` | `tests/test_tts_pipeline.py` |
| `src/eval/tts_eval.py` | `tests/test_tts_eval.py` |
| `src/eval/tts_vote_analysis.py` | `tests/test_tts_vote_analysis.py` |
| `src/eval/voting_replay.py` | `tests/test_voting_replay.py` |
| `src/data/datasets/treebench_export.py` | `tests/test_treebench_export.py` |
| `experiments/run_model_benchmark.py` | `tests/test_run_comparison.py` |

### 2. Write the test before touching the source file

When asked to implement a TODO, your first action must be to write or
complete the test cases for that unit, then implement against them.

### 3. Fixtures over mocks where possible

Use small, deterministic in-memory fixtures (e.g. a 64×64 PIL Image, a
hand-crafted list of candidate dicts) instead of patching external services.
Reserve `unittest.mock` for I/O boundaries (model API calls, disk access).

### 4. Tests must be fast and isolated

- No real model downloads or GPU calls in unit tests.
- Mock model `generate` / `_call_model` to return deterministic outputs.
- Use `tmp_path` (pytest built-in fixture) for anything that writes to disk.

### 5. One class, one behaviour per test method

Name tests as `test_<what>_<condition>_<expected>`.  For example:
```python
def test_majority_vote_tie_returns_first_seen_answer(): ...
def test_accuracy_empty_input_returns_zero(): ...
def test_build_candidate_inputs_returns_nine_entries(): ...
```

### 6. Parametrize sweeps

```python
@pytest.mark.parametrize("n", [1, 5, 9])
def test_tts_pipeline_generates_n_candidates(n): ...
```

---

## Implementing a new component

```
Step 1 — Open the corresponding test file.
Step 2 — Fill in the TODO test stubs (or add new test methods).
Step 3 — Run pytest and confirm tests FAIL (Red).
Step 4 — Open the source file and implement the class/function.
Step 5 — Run pytest and confirm tests PASS (Green).
Step 6 — Refactor if needed, keep tests green.
Step 7 — Update the __init__.py registry if the component is factory-built.
```

---

## Adding a new dataset

1. Write `tests/test_<name>.py` — cover `__len__`, `__getitem__`, split filtering, and `max_samples` cap.
2. Create `src/data/datasets/<name>.py` inheriting `BaseDataset`.
3. Register it in `src/data/datasets/__init__.py`.
4. Add a YAML config in `configs/datasets/<name>.yaml`.

## Adding a new model

1. Write `tests/test_models.py::TestMyModel` — cover `generate` contract (return type,
   required keys, n-chain count), lazy loading, and any model-specific helpers.
2. Create `src/models/my_model.py` inheriting `BaseVisualCoTModel`.
3. Implement `_load()` (idempotent, lazy), `_run_chain()`, and `generate()`.
4. Export the class from `src/models/__init__.py`.
5. Add a YAML config in `configs/models/my_model.yaml`.

### Model chain dict contract

Every model's `generate()` must return a list of dicts with **at minimum** these keys:

| Key | Type | Notes |
|---|---|---|
| `"bbox_raw"` | `str \| None` | Raw bbox string (VisCoT) or `None` |
| `"coords"` | `list[float]` | Parsed `[x1, y1, x2, y2]` or `[]` if not applicable |
| `"answer"` | `str` | Final answer (may be `""` if max turns exhausted) |

Agentic models (DeepEyesV2) additionally return:

| Key | Type | Notes |
|---|---|---|
| `"cot_steps"` | `list[str]` | One entry per agentic turn |
| `"tool_results"` | `list[str]` | Captured stdout / error per code execution |

---

## Agentic tool-calling CoT models (DeepEyesV2 pattern)

DeepEyesV2 implements a **multi-turn agentic loop** where the model can call a
Python code-execution tool to inspect the image programmatically before answering.

### Architecture

```
generate(image, query, n=N)
  └── for _ in range(n):
        _run_chain(image, query, temperature, max_new_tokens)
          └── loop up to max_turns:
                _call_model(messages) → response
                  ├── <answer>...</answer> found  → terminate, return answer
                  ├── <code>...</code> found       → _execute_code(), append tool result
                  └── neither                      → treat full response as answer, terminate
```

### Key constants

- `DEFAULT_MODEL_ID = "honglyhly/DeepEyesV2_7B_1031"`
- `DEFAULT_MAX_TURNS = 10`

### Code execution sandbox (`_execute_code`)

- Isolated `exec` namespace per chain — never shared between chains.
- Pre-populated with `image_1` (PIL Image, matching `PIL.Image.open()` semantics), `np`, `PIL`, `math`, `collections`.
- Captures stdout via `contextlib.redirect_stdout`; returns error summary on exception.
- Do **not** allow `matplotlib`, file writes, or network calls inside the sandbox.

### Parser helpers (module-level, public for testability)

| Function | Purpose |
|---|---|
| `_parse_answer(text)` | Extract `<answer>...</answer>` or `None` |
| `_extract_code_block(text)` | Extract `<code>...</code>` or `None` |
| `_execute_code(code, namespace)` | Run code, return stdout or `"ExcType: msg"` |

---

## Running tests

```bash
# All tests with coverage
pytest

# Single file
pytest tests/test_tts_pipeline.py -v

# Only fast unit tests
pytest -m "not integration"
```

---

## Compute: Narval (Digital Research Alliance of Canada)

All GPU experiments run on **Narval** (A100-40GB nodes).

### Account details
- **Username:** `dchikhi`
- **SLURM accounts:** `def-azouaq` (default), `aip-azouaq` (research project — prefer for GPU jobs)
- **Login node:** `narval.alliancecan.ca`

### Key paths on Narval
| Path | Use |
|---|---|
| `$HOME` (~50 GB) | Code, virtualenv |
| `$SCRATCH` (20 TB, purged after 60 days) | Datasets, HuggingFace cache, results |
| `$SLURM_TMPDIR` (fast local SSD, cleared after job) | Copy data here at job start for fast I/O |

### HuggingFace model cache
Compute nodes have **no internet access**. The model must be downloaded on the login node first:
```bash
# On login node only — set cache to scratch so it survives
export HF_HOME=$SCRATCH/hf_cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('honglyhly/DeepEyesV2_7B_1031')"
```
Always set `HF_HOME=$SCRATCH/hf_cache` in job scripts so the model is found.

### Environment setup (run once on login node)
```bash
module load python/3.11 cuda/12.2
python -m venv $HOME/envs/viscot
source $HOME/envs/viscot/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install 'transformers>=4.45.0' accelerate qwen-vl-utils bitsandbytes
pip install -e .   # install this repo
```

### Example SLURM job script
```bash
#!/bin/bash
#SBATCH --account=aip-azouaq
#SBATCH --job-name=deepeyes_vqa
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

module load python/3.11 cuda/12.2
source $HOME/envs/viscot/bin/activate

export HF_HOME=$SCRATCH/hf_cache

python experiments/run_model_benchmark.py --config configs/experiments/baseline.yaml
```

### Writing new job scripts
- Always use `--account=aip-azouaq` for research GPU jobs.
- Request `--mem=40G` for one A100 (matches VRAM); add more for large batches.
- Put output files under `$SCRATCH/results/`, never `$HOME`.
- Copy dataset to `$SLURM_TMPDIR` at job start if doing many random reads.

---

## Experiment reproducibility

- Experiments are driven by YAML configs under `configs/experiments/`.
- Results are saved to `results/<run_name>/` with `predictions.jsonl` and `metrics.json`.
- Never hard-code output paths in source files.

---

## Code style

- Python 3.10+, type hints everywhere.
- Line length: 100 characters (`black` + `ruff` enforced via `pyproject.toml`).
- Docstrings: Google style for all public classes and functions.
- No bare `except:` — always catch a specific exception type.

---

## What NOT to do

- Do not write implementation code before its tests exist.
- Do not merge a PR where `pytest` reports failures.
- Do not add `print()` statements to production code — use `get_logger()`.
- Do not hard-code model names, paths, or hyperparameters — use configs.
- Do not commit large binary files (images, checkpoints) — use remote storage.
- Do not share execution namespaces between agentic chains.
- Do not allow the code sandbox to import `matplotlib`, perform file I/O, or make network requests.
- Do not raise an exception when `max_turns` is exhausted — return `answer = ""`.
