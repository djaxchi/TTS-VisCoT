"""Tests for alternative voting strategies.

Each strategy takes a list of candidate answers and a greedy answer,
and returns the selected answer.
"""

import pytest

from scripts.voting_strategies import (
    vote_plurality,
    vote_greedy_tiebreak,
    vote_greedy_unless_supermajority,
    vote_consistency_filter,
    evaluate_strategies,
)


# ── Plurality (baseline) ───────────────────────────────────────────────────

class TestPlurality:

    def test_clear_winner(self):
        assert vote_plurality(["A", "A", "A", "B", "C"], greedy="B") == "A"

    def test_tie_returns_first_seen(self):
        # A appears at idx 0,1; B at idx 2,3 — A wins (first seen)
        assert vote_plurality(["A", "A", "B", "B", "C"], greedy="X") == "A"

    def test_single_candidate(self):
        assert vote_plurality(["A"], greedy="A") == "A"

    def test_all_different(self):
        assert vote_plurality(["A", "B", "C"], greedy="X") == "A"

    def test_skips_none(self):
        assert vote_plurality(["A", None, "A", "B", None], greedy="B") == "A"

    def test_all_none_returns_none(self):
        assert vote_plurality([None, None], greedy="A") is None


# ── Greedy tiebreak ────────────────────────────────────────────────────────

class TestGreedyTiebreak:

    def test_clear_winner_ignores_greedy(self):
        assert vote_greedy_tiebreak(["A", "A", "A", "B"], greedy="B") == "A"

    def test_tie_resolved_by_greedy(self):
        # A:2, B:2 — greedy is B, so B wins
        assert vote_greedy_tiebreak(["A", "A", "B", "B"], greedy="B") == "B"

    def test_tie_greedy_not_in_tie_falls_back_to_first(self):
        # A:2, B:2, greedy is C — C not in tie, fall back to first seen
        assert vote_greedy_tiebreak(["A", "A", "B", "B"], greedy="C") == "A"


# ── Greedy-unless-supermajority ────────────────────────────────────────────

class TestGreedyUnlessSupermajority:

    def test_keeps_greedy_when_no_supermajority(self):
        # 3 out of 9 say B — not enough to override greedy A
        candidates = ["A", "B", "B", "B", "C", "D", "E", "F", "A"]
        assert vote_greedy_unless_supermajority(candidates, greedy="A", threshold=5) == "A"

    def test_overrides_greedy_when_supermajority(self):
        # 5 out of 9 say B — supermajority overrides greedy A
        candidates = ["A", "B", "B", "B", "B", "B", "C", "D", "A"]
        assert vote_greedy_unless_supermajority(candidates, greedy="A", threshold=5) == "B"

    def test_keeps_greedy_at_exact_threshold_minus_one(self):
        candidates = ["B", "B", "B", "B", "A", "A", "C", "D", "E"]
        assert vote_greedy_unless_supermajority(candidates, greedy="A", threshold=5) == "A"

    def test_overrides_at_exact_threshold(self):
        candidates = ["B", "B", "B", "B", "B", "A", "C", "D", "E"]
        assert vote_greedy_unless_supermajority(candidates, greedy="A", threshold=5) == "B"

    def test_greedy_none_falls_back_to_plurality(self):
        candidates = ["A", "A", "A", "B", "B"]
        assert vote_greedy_unless_supermajority(candidates, greedy=None, threshold=3) == "A"

    def test_greedy_matches_supermajority(self):
        candidates = ["A", "A", "A", "A", "A", "B", "C", "D", "E"]
        assert vote_greedy_unless_supermajority(candidates, greedy="A", threshold=5) == "A"


# ── Consistency filter ─────────────────────────────────────────────────────

class TestConsistencyFilter:

    def test_filters_singletons(self):
        # A:3, B:1, C:1 → only A survives → A wins
        assert vote_consistency_filter(["A", "A", "A", "B", "C"], greedy="B",
                                       min_count=2) == "A"

    def test_multiple_survivors(self):
        # A:3, B:2, C:1 → A and B survive → A wins by plurality
        assert vote_consistency_filter(["A", "A", "A", "B", "B", "C"], greedy="C",
                                       min_count=2) == "A"

    def test_all_singletons_falls_back_to_greedy(self):
        assert vote_consistency_filter(["A", "B", "C", "D", "E"], greedy="X",
                                       min_count=2) == "X"

    def test_greedy_none_and_all_singletons_falls_back_to_plurality(self):
        assert vote_consistency_filter(["A", "B", "C"], greedy=None,
                                       min_count=2) == "A"


# ── Evaluate strategies ────────────────────────────────────────────────────

class TestEvaluateStrategies:

    def _make_row(self, gt: str, greedy_ans: str, all_answers: list[str | None]) -> dict:
        candidates = []
        for i, ans in enumerate(all_answers):
            candidates.append({
                "candidate_idx": i,
                "image_aug": "original" if i == 0 else "edge_enhance",
                "text_variant": "original",
                "temperature": 0.0 if i == 0 else 0.7,
                "answer": ans,
            })
        return {
            "task": "vqa",
            "gt_answer": gt,
            "answers_all": [gt],
            "candidates": candidates,
            "greedy": greedy_ans,
        }

    def test_returns_accuracy_per_strategy(self):
        rows = [
            self._make_row("A", "A", ["A", "B", "B", "B"]),  # greedy correct, plurality wrong
            self._make_row("B", "C", ["C", "B", "B", "B"]),  # greedy wrong, plurality correct
        ]
        results = evaluate_strategies(rows)
        assert "plurality" in results
        assert "greedy" in results
        assert results["greedy"]["correct"] == 1
        assert results["greedy"]["total"] == 2

    def test_oracle_computed(self):
        rows = [
            self._make_row("A", "B", ["B", "A", "C", "D"]),  # A present → oracle correct
            self._make_row("X", "Y", ["Y", "Z", "W", "Q"]),  # X absent → oracle wrong
        ]
        results = evaluate_strategies(rows)
        assert results["oracle"]["correct"] == 1
        assert results["oracle"]["total"] == 2
