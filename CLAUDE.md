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
├── configs/         YAML experiment configs (datasets, models, experiments)
├── experiments/     Top-level runnable scripts (run_baseline.py, run_tts.py, ablations/)
├── notebooks/       Exploratory analysis and results visualisation
├── results/         Output artefacts — gitignored except .gitkeep
├── scripts/         Shell utilities (setup_env.sh, run_ablations.sh)
├── src/
│   ├── data/        Dataset loaders + augmentation strategies
│   ├── eval/        Metrics, benchmark runner, visualisations
│   ├── methods/     baseline.py and tts/ (sampling + scaling)
│   ├── models/      BaseVisualCoTModel + VisualCoTModel (HuggingFace)
│   ├── utils/       logging, I/O helpers
│   └── voting/      Aggregation systems (majority, weighted, best_of_n)
└── tests/           Mirrors src/ structure; one test file per module
```

---

## TDD rules for this project

### 1. Tests live in `tests/`, mirrors `src/`

| Source file | Test file |
|---|---|
| `src/voting/majority.py` | `tests/test_voting.py` → `TestMajorityVote` |
| `src/eval/metrics.py` | `tests/test_eval.py` → `TestComputeMetrics` |
| `src/data/augmentation/geometric.py` | `tests/test_data.py` → `TestGeometricAugmentation` |

### 2. Write the test before touching the source file

When asked to implement a TODO, your first action must be to write or
complete the test cases for that unit, then implement against them.

### 3. Fixtures over mocks where possible

Use small, deterministic in-memory fixtures (e.g. a 64×64 PIL Image, a
hand-crafted list of chain dicts) instead of patching external services.
Reserve `unittest.mock` for I/O boundaries (model API calls, disk access).

### 4. Tests must be fast and isolated

- No real model downloads or GPU calls in unit tests.
- Mock `BaseVisualCoTModel.generate` to return deterministic chain lists.
- Use `tmp_path` (pytest built-in fixture) for anything that writes to disk.

### 5. One class, one behaviour per test method

Name tests as `test_<what>_<condition>_<expected>`.  For example:
```python
def test_majority_vote_tie_returns_first_seen_answer(): ...
def test_accuracy_empty_input_returns_zero(): ...
def test_geometric_aug_output_size_equals_input_size(): ...
```

### 6. Parametrize sweeps

Use `@pytest.mark.parametrize` to cover edge cases without duplicating test bodies:
```python
@pytest.mark.parametrize("n", [1, 4, 8, 16, 32])
def test_tts_method_calls_generate_exactly_n_times(n): ...
```

---

## Implementing a new component

Follow this sequence every time:

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

1. Write `tests/test_data.py::TestMyDataset` — cover `__len__`, `__getitem__`,
   `split` filtering, and `max_samples` cap.
2. Create `src/data/datasets/my_dataset.py` inheriting `BaseDataset`.
3. Register it in `src/data/datasets/__init__.py::_DATASET_REGISTRY`.
4. Add a YAML config in `configs/datasets/my_dataset.yaml`.

## Adding a new augmentation

1. Write `tests/test_data.py::TestMyAugmentation`.
2. Create `src/data/augmentation/my_aug.py` inheriting `BaseAugmentation`.
3. Register it in `src/data/augmentation/__init__.py::_AUGMENTATION_REGISTRY`.

## Adding a new voting system

1. Write `tests/test_voting.py::TestMyVoting` using the shared `CHAINS_*` fixtures.
2. Create `src/voting/my_voting.py` inheriting `BaseVotingSystem`.
3. Register it in `src/voting/__init__.py::_VOTING_REGISTRY`.

---

## Running tests

```bash
# All tests with coverage
pytest

# Single module
pytest tests/test_voting.py -v

# Only fast unit tests (no integration)
pytest -m "not integration"

# Watch mode (requires pytest-watch)
ptw tests/
```

Coverage target: **≥ 90 %** on `src/` before any experiment is considered valid.

---

## Experiment reproducibility

- Every experiment must be driven by a YAML config under `configs/experiments/`.
- Random seeds must be set in the config (`seed` key) and respected by all
  stochastic components (augmentation, model sampling).
- Results are saved to `results/<run_name>/` with `predictions.jsonl` and
  `metrics.json`.  Never hard-code output paths in source files.

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
