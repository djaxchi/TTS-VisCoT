# TTS-VisCoT: Test-Time Scaling for Visual Chain-of-Thought

> **Research repository** — Investigating how test-time compute scaling enhances the reasoning capability of Visual Chain-of-Thought (CoT) models on visual cognitive tasks.

---

## Overview

This project benchmarks **test-time scaling (TTS)** strategies against baselines on visual chain-of-thought tasks. We evaluate across:

- **Datasets** — Standardized visual reasoning benchmarks (initially scoped to counting tasks)
- **Data augmentation** — Augmentation pipelines applied at inference time to diversify the reasoning paths sampled
- **Voting / aggregation systems** — Methods for combining multiple sampled CoT paths into a final prediction (majority vote, weighted vote, best-of-N, etc.)

The central hypothesis is that allowing a visual CoT model more compute at test time — through repeated sampling + smart aggregation — yields measurable accuracy gains, even without additional training.

---

## Repository Structure

```
TTS-VisCoT/
├── configs/                    # YAML configuration files
│   ├── datasets/               #   per-dataset configs
│   ├── models/                 #   model / backbone configs
│   └── experiments/            #   full experiment configs (baseline, TTS variants)
│
├── src/                        # Core source package
│   ├── data/
│   │   ├── datasets/           #   Dataset loaders (counting, …)
│   │   └── augmentation/       #   Augmentation strategies (geometric, semantic, …)
│   ├── models/                 #   Model wrappers / Visual-CoT interface
│   ├── methods/
│   │   ├── baseline.py         #   Greedy / single-pass baseline
│   │   └── tts/                #   Test-time scaling strategies (sampling, budgeting)
│   ├── voting/                 #   Aggregation / voting systems
│   ├── eval/                   #   Metrics, benchmark runner, result visualizations
│   └── utils/                  #   Logging, I/O helpers
│
├── experiments/                # Top-level runnable experiment scripts
│   ├── run_baseline.py
│   ├── run_tts.py
│   └── ablations/              #   Isolated ablation scripts
│
├── notebooks/                  # Exploratory & results analysis notebooks
│
├── results/                    # Output artefacts (ignored by git except structure)
│
├── scripts/                    # Shell utility scripts (env setup, data download)
│
└── tests/                      # Unit & integration tests
```

---

## Key Concepts

| Term | Definition |
|---|---|
| **Visual CoT** | A multimodal LLM prompted to produce a step-by-step reasoning chain before giving a final answer to a visual question |
| **Test-time scaling (TTS)** | Allocating additional compute at inference time (e.g., sampling N completions) without changing model weights |
| **Voting system** | An aggregation function that maps N candidate answers/chains to a single final prediction |
| **Augmentation** | Transformations applied to the input image (and/or prompt) to increase diversity across sampled reasoning chains |

---

## Experimental Axes

### 1. Datasets
- `counting` — Visual counting tasks (primary benchmark)
- *(extensible)* — Additional visual reasoning categories

### 2. Data Augmentation Methods
| Method | Description |
|---|---|
| `none` | No augmentation (baseline) |
| `geometric` | Flips, crops, rotations, color jitter |
| `semantic` | Caption-guided or region-mask perturbations |
| `mixed` | Combination of geometric + semantic |

### 3. Voting / Aggregation Systems
| System | Description |
|---|---|
| `majority` | Plurality vote over final predicted answers |
| `weighted` | Vote weighted by model confidence / log-prob |
| `best_of_n` | Select chain with highest self-consistency score |
| `orm` | Outcome reward model re-ranking |

### 4. Scaling Budgets (N)
`N ∈ {1, 4, 8, 16, 32, 64}` — number of sampled completions per example.

---

## Getting Started

### Installation

```bash
git clone https://github.com/djaxchi/TTS-VisCoT.git
cd TTS-VisCoT
pip install -e ".[dev]"
```
### Test model

cd /Users/djadja/Code/TTS-VisCoT && /opt/homebrew/bin/python3.11 experiments/run_viscot.py \
    --image "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1200px-Cute_dog.jpg" \
    --query "What is the dog doing?" 2>&1

### Running a Baseline Experiment

```bash
python experiments/run_baseline.py \
    --config configs/experiments/baseline.yaml \
    --dataset counting \
    --output results/baseline_counting/
```

### Running a TTS Experiment

```bash
python experiments/run_tts.py \
    --config configs/experiments/tts_majority_vote.yaml \
    --dataset counting \
    --n_samples 16 \
    --voting majority \
    --augmentation geometric \
    --output results/tts_counting_n16/
```

### Running All Ablations

```bash
bash scripts/run_ablations.sh
```

---

## Results

Results and analysis notebooks are stored in `results/` and `notebooks/`. Once experiments are complete, summary tables and plots will be populated here.

| Method | Augmentation | Voting | N | Accuracy |
|---|---|---|---|---|
| Baseline | none | — | 1 | — |
| TTS | none | majority | 16 | — |
| TTS | geometric | majority | 16 | — |
| TTS | geometric | weighted | 16 | — |
| TTS | semantic | majority | 16 | — |

---

## Contributing

This is an active research project. See `docs/contributing.md` for conventions on adding new datasets, augmentation methods, or voting strategies.

---

## Citation

*(To be added upon publication)*

---

## License

MIT



python experiments/run_comparison.py --n 20 --resume --save-output results/comparison/run2.json