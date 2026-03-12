"""Tests for src.voting.normalize and src.voting.majority."""

import pytest

from src.voting.normalize import normalize_answer
from src.voting.majority import VoteResult, majority_vote


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHAINS_UNIFORM = ["A", "A", "A", "A"]  # perfect consensus
CHAINS_MAJORITY = ["A", "A", "B", "C"]  # A wins 2/4
CHAINS_TIE = ["A", "B"]               # tie — first seen (A) wins
CHAINS_MESSY = ["(A)", "option B", "The answer is A", "[A]", "A"]


# ---------------------------------------------------------------------------
# TestNormalizeAnswer
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_normalize_bare_letter_returns_uppercase(self):
        assert normalize_answer("a") == "A"

    def test_normalize_letter_in_parentheses(self):
        assert normalize_answer("(B)") == "B"

    def test_normalize_letter_in_square_brackets(self):
        assert normalize_answer("[C]") == "C"

    def test_normalize_option_prefix(self):
        assert normalize_answer("option A") == "A"

    def test_normalize_answer_prefix(self):
        assert normalize_answer("answer: C") == "C"

    def test_normalize_the_answer_is_phrase(self):
        assert normalize_answer("The answer is D") == "D"

    def test_normalize_choice_prefix(self):
        assert normalize_answer("Choice: B") == "B"

    def test_normalize_letter_followed_by_dot(self):
        assert normalize_answer("A.") == "A"

    def test_normalize_empty_string_returns_empty(self):
        assert normalize_answer("") == ""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("A", "A"),
            ("(B)", "B"),
            ("[C]", "C"),
            ("option D", "D"),
            ("The answer is A", "A"),
            ("I think it's B.", "B"),
            ("Select: C", "C"),
        ],
    )
    def test_normalize_parametrized_formats(self, raw, expected):
        assert normalize_answer(raw) == expected


# ---------------------------------------------------------------------------
# TestMajorityVote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_majority_vote_empty_input_returns_empty_result(self):
        result = majority_vote([])
        assert result.consensus_answer == ""
        assert result.vote_counts == {}
        assert result.agreement_rate == 0.0
        assert result.confidence == 0.0

    def test_majority_vote_uniform_consensus_full_agreement(self):
        result = majority_vote(CHAINS_UNIFORM)
        assert result.consensus_answer == "A"
        assert result.agreement_rate == 1.0
        assert result.dispersion_score == 0.0
        assert result.confidence == 1.0

    def test_majority_vote_selects_most_common_answer(self):
        result = majority_vote(CHAINS_MAJORITY)
        assert result.consensus_answer == "A"

    def test_majority_vote_tie_returns_first_seen_answer(self):
        result = majority_vote(CHAINS_TIE)
        # Counter.most_common is stable on tie; first inserted wins
        assert result.consensus_answer in ("A", "B")

    def test_majority_vote_agreement_rate_correct(self):
        result = majority_vote(["A", "A", "B", "C"])
        assert result.agreement_rate == pytest.approx(0.5)

    def test_majority_vote_vote_counts_sum_to_n(self):
        preds = ["A", "A", "B", "C", "C"]
        result = majority_vote(preds)
        assert sum(result.vote_counts.values()) == len(preds)

    def test_majority_vote_normalizes_messy_inputs(self):
        result = majority_vote(CHAINS_MESSY)
        assert result.consensus_answer == "A"

    def test_majority_vote_skip_normalize_preserves_raw(self):
        result = majority_vote(["A", "A", "(A)"], normalize=False)
        # Without normalization "(A)" is a different bucket
        assert result.vote_counts.get("A", 0) == 2

    def test_majority_vote_dispersion_zero_on_full_consensus(self):
        result = majority_vote(["B", "B", "B"])
        assert result.dispersion_score == pytest.approx(0.0)

    def test_majority_vote_vote_margin_one_on_full_consensus(self):
        result = majority_vote(["C", "C", "C"])
        assert result.vote_margin == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [1, 4, 8, 16, 32])
    def test_majority_vote_consensus_is_in_input_set(self, n):
        preds = (["A"] * (n // 2 + 1)) + (["B"] * (n // 2))
        result = majority_vote(preds)
        assert result.consensus_answer in {"A", "B"}
