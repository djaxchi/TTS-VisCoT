"""Tests for token-trace analytics helpers."""

from __future__ import annotations

from src.eval.token_trace import (
    token_lengths,
    token_majority_sequence,
    token_position_agreement,
)


class TestTokenLengths:
    def test_token_lengths_returns_length_per_candidate(self) -> None:
        rows = [["a", "b"], ["c"], []]
        assert token_lengths(rows) == [2, 1, 0]


class TestTokenPositionAgreement:
    def test_agreement_computes_majority_fraction_per_position(self) -> None:
        rows = [
            ["The", "Ġbird", "Ġis"],
            ["The", "Ġbird", "Ġis"],
            ["The", "Ġcat", "Ġis"],
        ]
        agr = token_position_agreement(rows)
        assert agr == [1.0, 2 / 3, 1.0]

    def test_agreement_handles_empty_input(self) -> None:
        assert token_position_agreement([]) == []


class TestTokenMajoritySequence:
    def test_majority_sequence_picks_first_seen_on_tie(self) -> None:
        rows = [
            ["a", "x"],
            ["b", "x"],
        ]
        seq = token_majority_sequence(rows)
        # position 0 tie -> first seen "a"
        assert seq[0] == "a"
        assert seq[1] == "x"

    def test_majority_sequence_strips_qwen_space_marker(self) -> None:
        rows = [["Ġ10", "Ġyears"], ["10", "years"]]
        assert token_majority_sequence(rows) == ["10", "years"]
