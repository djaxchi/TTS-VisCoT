"""Tests for replaying voting methods on saved candidate traces."""

from __future__ import annotations

import pytest

from src.eval.voting_replay import (
    compute_reliability_weights,
    replay_method_answer,
)


def _entries() -> list[dict]:
    return [
        {
            "references": ["cat"],
            "candidate_answers_normalized": ["dog", "cat", "cat", "cat", "cat"],
            "voting": {
                "majority_3": {"answer": "cat"},
                "majority_5": {"answer": "cat", "agreement_rate": 0.8},
            },
        },
        {
            "references": ["8"],
            "candidate_answers_normalized": ["8", "8", "7", "7", "7"],
            "voting": {
                "majority_3": {"answer": "8"},
                "majority_5": {"answer": "7", "agreement_rate": 0.6},
            },
        },
    ]


class TestReplayMethodAnswer:
    def test_zero_shot_returns_first_candidate(self) -> None:
        e = _entries()[0]
        assert replay_method_answer(e, method="zero_shot") == "dog"

    def test_majority_3_uses_saved_vote(self) -> None:
        e = _entries()[1]
        assert replay_method_answer(e, method="majority_3") == "8"

    def test_majority_5_uses_saved_vote(self) -> None:
        e = _entries()[1]
        assert replay_method_answer(e, method="majority_5") == "7"

    def test_weighted_slot_prefers_high_weight_correct_answer(self) -> None:
        e = _entries()[1]
        # Heavy weight on first two candidates should recover "8"
        ans = replay_method_answer(
            e,
            method="weighted_slot",
            weights=[0.4, 0.35, 0.1, 0.1, 0.05],
        )
        assert ans == "8"

    def test_token_majority_builds_consensus_phrase(self) -> None:
        e = {
            "references": ["white parrot"],
            "candidate_answers_normalized": [
                "white parrot",
                "white cockatoo",
                "white parrot",
                "white parrot",
                "white cockatoo",
            ],
            "voting": {
                "majority_3": {"answer": "white parrot"},
                "majority_5": {"answer": "white parrot", "agreement_rate": 0.6},
            },
        }
        ans = replay_method_answer(e, method="token_majority")
        assert ans == "white parrot"

    def test_token_majority_returns_empty_for_empty_candidates(self) -> None:
        e = {
            "references": ["cat"],
            "candidate_answers_normalized": ["", "", ""],
            "voting": {},
        }
        ans = replay_method_answer(e, method="token_majority")
        assert ans == ""

    def test_gated_majority_falls_back_when_agreement_low(self) -> None:
        e = _entries()[1]
        ans = replay_method_answer(
            e,
            method="gated_majority_5",
            threshold=0.8,
        )
        assert ans == "8"  # fallback to zero-shot


class TestComputeReliabilityWeights:
    def test_weights_sum_to_one(self) -> None:
        w = compute_reliability_weights(_entries())
        assert sum(w) == pytest.approx(1.0)

    def test_weights_length_is_five(self) -> None:
        w = compute_reliability_weights(_entries())
        assert len(w) == 5

    def test_better_slots_get_higher_weight(self) -> None:
        w = compute_reliability_weights(_entries())
        # In fixture, slot2 is always correct, slot4/5 only half the time.
        assert w[1] > w[3]
        assert w[1] > w[4]
