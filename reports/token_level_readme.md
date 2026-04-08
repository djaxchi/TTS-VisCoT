# Token-Level Aggregation Prototype

This note describes the token-level aggregation prototype added in parallel to answer-level voting.

## What it does

- Uses the same candidate inputs (image + prompt variants).
- Decodes in synchronized steps across all candidates.
- At each step:
  - collects next-token logits per candidate,
  - averages logits,
  - picks one shared token,
  - appends it to every candidate context.
- Stops when a normalized A/B/C/D answer is detected or max steps are reached.

## Files

- `src/token_aggregation.py`
  - `aggregate_answer_level(...)`
  - `aggregate_token_level(...)`
- `src/check_token_support.py`
  - feasibility report builder
- `scripts/smoke_token_aggregation.py`
  - one-example smoke script
- `reports/token_level_feasibility.md`
  - feasibility summary for current backends

## Run feasibility report

```bash
python src/check_token_support.py
```

## Run smoke test (one exported sample)

```bash
python scripts/smoke_token_aggregation.py \
  --exported-dir results/treebench_samples \
  --index 0
```

## Notes

- Current prototype targets Qwen-based backends first (DirectVLM / GRIT backbone).
- VisualCoT backend requires additional backend-specific refactoring due to its 2-turn crop flow.
- Existing answer-level pipeline remains unchanged.
