"""Tests for token-level aggregation prototype."""

from __future__ import annotations

import numpy as np
import pytest

from src.token_aggregation import (
    aggregate_answer_level,
    aggregate_token_level_from_logits_steps,
    top_k_token_distribution,
)


def test_aggregate_answer_level_majority_vote() -> None:
    answers = ["A", "B", "B", "C", "B"]
    out = aggregate_answer_level(answers)
    assert out["winning_answer"] == "B"
    assert out["vote_counts"]["B"] == 3


def test_token_level_average_logits_selects_shared_token() -> None:
    # 2 candidates, 3-token vocab; both favor token 1 on step 1.
    step_logits = [
        np.array([[0.1, 3.0, 0.2], [0.0, 2.5, 0.1]], dtype=float),
    ]
    out = aggregate_token_level_from_logits_steps(
        step_logits,
        token_id_to_text={0: "A", 1: "B", 2: "C"},
        max_steps=1,
    )
    assert out["chosen_token_ids"] == [1]
    assert out["generated_text"] == "B"


def test_token_level_restrict_choices_abcd() -> None:
    # Unconstrained max would be token 4; constrained should pick among A/B/C/D ids.
    step_logits = [
        np.array([[0.1, 0.2, 0.3, 0.4, 9.0], [0.2, 0.1, 0.2, 0.5, 8.0]], dtype=float),
    ]
    out = aggregate_token_level_from_logits_steps(
        step_logits,
        token_id_to_text={0: "A", 1: "B", 2: "C", 3: "D", 4: "X"},
        choice_token_ids=[0, 1, 2, 3],
        max_steps=1,
    )
    assert out["chosen_token_ids"][0] in {0, 1, 2, 3}


def test_token_level_stops_when_normalized_answer_found() -> None:
    step_logits = [
        np.array([[9.0, 0.1, 0.1, 0.1], [8.5, 0.1, 0.1, 0.1]], dtype=float),
        np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=float),
    ]
    out = aggregate_token_level_from_logits_steps(
        step_logits,
        token_id_to_text={0: "A", 1: "B", 2: "C", 3: "D"},
        max_steps=5,
    )
    assert out["normalized_answer"] == "A"
    assert len(out["chosen_token_ids"]) == 1


# ---------------------------------------------------------------------------
# top_k_token_distribution tests
# ---------------------------------------------------------------------------

def test_top_k_token_distribution_returns_k_entries() -> None:
    logits = np.array([1.0, 2.0, 3.0, 0.5, 0.1], dtype=float)
    out = top_k_token_distribution(logits, k=3, token_id_to_text={i: f"T{i}" for i in range(5)})
    assert len(out) == 3


def test_top_k_token_distribution_sorted_descending_prob() -> None:
    logits = np.array([1.0, 2.0, 3.0, 0.5, 0.1], dtype=float)
    out = top_k_token_distribution(logits, k=5, token_id_to_text={i: f"T{i}" for i in range(5)})
    probs = [e["prob"] for e in out]
    assert probs == sorted(probs, reverse=True)


def test_top_k_token_distribution_full_vocab_probs_sum_to_one() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=float)
    out = top_k_token_distribution(logits, k=3, token_id_to_text={0: "A", 1: "B", 2: "C"})
    assert abs(sum(e["prob"] for e in out) - 1.0) < 1e-6


def test_top_k_token_distribution_log_prob_consistent_with_prob() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=float)
    out = top_k_token_distribution(logits, k=3, token_id_to_text={0: "A", 1: "B", 2: "C"})
    for e in out:
        assert abs(np.exp(e["log_prob"]) - e["prob"]) < 1e-6


def test_top_k_token_distribution_top1_is_argmax() -> None:
    logits = np.array([0.1, 0.2, 5.0, 0.3], dtype=float)
    out = top_k_token_distribution(logits, k=2, token_id_to_text={i: f"T{i}" for i in range(4)})
    assert out[0]["token_id"] == 2


def test_top_k_token_distribution_required_keys() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=float)
    out = top_k_token_distribution(logits, k=2, token_id_to_text={0: "A", 1: "B", 2: "C"})
    for entry in out:
        assert set(entry.keys()) >= {"token_id", "token_text", "logit", "prob", "log_prob"}


def test_top_k_token_distribution_unknown_token_id_uses_empty_text() -> None:
    logits = np.array([1.0, 5.0, 2.0], dtype=float)
    out = top_k_token_distribution(logits, k=1, token_id_to_text={0: "A"})
    # token_id 1 has highest logit but is not in token_id_to_text
    assert out[0]["token_id"] == 1
    assert out[0]["token_text"] == ""


@pytest.mark.parametrize("k", [1, 2, 5])
def test_top_k_token_distribution_k_clipped_to_vocab_size(k: int) -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=float)  # vocab_size=3
    out = top_k_token_distribution(logits, k=k, token_id_to_text={0: "A", 1: "B", 2: "C"})
    assert len(out) == min(k, 3)


# ---------------------------------------------------------------------------
# top_k integration with aggregate_token_level_from_logits_steps
# ---------------------------------------------------------------------------

def test_aggregate_logits_steps_top_k_present_in_steps() -> None:
    step_logits = [np.array([[0.1, 3.0, 0.2], [0.0, 2.5, 0.1]], dtype=float)]
    out = aggregate_token_level_from_logits_steps(
        step_logits,
        token_id_to_text={0: "A", 1: "B", 2: "C"},
        max_steps=1,
        top_k=2,
    )
    assert "top_k" in out["steps"][0]
    assert len(out["steps"][0]["top_k"]) == 2
    assert "prob" in out["steps"][0]["top_k"][0]
    assert "log_prob" in out["steps"][0]["top_k"][0]


def test_aggregate_logits_steps_top_k_zero_omits_key() -> None:
    step_logits = [np.array([[0.1, 3.0, 0.2], [0.0, 2.5, 0.1]], dtype=float)]
    out = aggregate_token_level_from_logits_steps(
        step_logits,
        token_id_to_text={0: "A", 1: "B", 2: "C"},
        max_steps=1,
    )
    assert "top_k" not in out["steps"][0]


def test_aggregate_logits_steps_top_k_best_token_matches_chosen() -> None:
    """The top-1 entry in top_k must match the chosen token."""
    step_logits = [np.array([[0.1, 3.0, 0.2], [0.0, 2.5, 0.1]], dtype=float)]
    out = aggregate_token_level_from_logits_steps(
        step_logits,
        token_id_to_text={0: "A", 1: "B", 2: "C"},
        max_steps=1,
        top_k=3,
    )
    step = out["steps"][0]
    assert step["top_k"][0]["token_id"] == step["chosen_token_id"]
