"""Tests for src.models and src.methods (using mocked model)."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.models.base import BaseVisualCoTModel
from src.methods.baseline import run_baseline
from src.methods.tts.sampling import run_tts_sampling
from src.methods.tts.scaling import run_tts_scaling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_IMAGE = Image.new("RGB", (64, 64), color=(10, 20, 30))
QUERY = "What is shown in the image?"
OPTIONS = {"A": "Cat", "B": "Dog", "C": "Fish", "D": "Bird"}


def _make_chain(answer: str = "A") -> Dict[str, Any]:
    return {"bbox_raw": "[0.1, 0.1, 0.9, 0.9]", "coords": [0.1, 0.1, 0.9, 0.9], "answer": answer}


class _MockModel(BaseVisualCoTModel):
    """Deterministic mock: always returns the same chain."""

    def __init__(self, answer: str = "A"):
        self._answer = answer
        self.call_count = 0

    def generate(self, image, query, *, n=1, temperature=0.0, max_new_tokens=512, **kw):
        self.call_count += n
        return [_make_chain(self._answer) for _ in range(n)]


# ---------------------------------------------------------------------------
# TestBaseVisualCoTModel
# ---------------------------------------------------------------------------


class TestBaseVisualCoTModel:
    def test_predict_calls_generate_with_n_1(self):
        model = _MockModel("B")
        result = model.predict(TINY_IMAGE, QUERY)
        assert model.call_count == 1
        assert result["answer"] == "B"

    def test_predict_returns_single_dict(self):
        model = _MockModel("C")
        result = model.predict(TINY_IMAGE, QUERY)
        assert isinstance(result, dict)

    def test_generate_returns_list_of_n_chains(self):
        model = _MockModel("A")
        chains = model.generate(TINY_IMAGE, QUERY, n=5)
        assert len(chains) == 5


# ---------------------------------------------------------------------------
# TestRunBaseline
# ---------------------------------------------------------------------------


class TestRunBaseline:
    def test_run_baseline_returns_chain_dict(self):
        model = _MockModel("D")
        result = run_baseline(model, TINY_IMAGE, QUERY)
        assert "answer" in result

    def test_run_baseline_calls_model_once(self):
        model = _MockModel("A")
        run_baseline(model, TINY_IMAGE, QUERY)
        assert model.call_count == 1


# ---------------------------------------------------------------------------
# TestTTSSampling
# ---------------------------------------------------------------------------


class TestTTSSampling:
    @pytest.mark.parametrize("n", [1, 4, 8, 16, 32])
    def test_tts_method_calls_generate_exactly_n_times(self, n):
        model = _MockModel("A")
        run_tts_sampling(model, TINY_IMAGE, QUERY, n=n)
        assert model.call_count == n

    def test_tts_sampling_returns_answer_key(self):
        model = _MockModel("B")
        result = run_tts_sampling(model, TINY_IMAGE, QUERY, n=3)
        assert "answer" in result

    def test_tts_sampling_answer_is_from_vote(self):
        model = _MockModel("C")
        result = run_tts_sampling(model, TINY_IMAGE, QUERY, n=5)
        assert result["answer"] == "C"

    def test_tts_sampling_returns_chains_list(self):
        model = _MockModel("A")
        result = run_tts_sampling(model, TINY_IMAGE, QUERY, n=4)
        assert len(result["chains"]) == 4

    def test_tts_sampling_vote_result_present(self):
        from src.voting.majority import VoteResult

        model = _MockModel("A")
        result = run_tts_sampling(model, TINY_IMAGE, QUERY, n=3)
        assert isinstance(result["vote_result"], VoteResult)


# ---------------------------------------------------------------------------
# TestTTSScaling
# ---------------------------------------------------------------------------


class TestTTSScaling:
    def test_tts_scaling_returns_answer_key(self):
        model = _MockModel("A")
        result = run_tts_scaling(model, TINY_IMAGE, QUERY, OPTIONS)
        assert "answer" in result

    def test_tts_scaling_returns_views_and_chains(self):
        model = _MockModel("B")
        result = run_tts_scaling(model, TINY_IMAGE, QUERY, OPTIONS)
        assert "views" in result
        assert "chains" in result
        assert len(result["views"]) == len(result["chains"])

    def test_tts_scaling_answer_matches_majority(self):
        model = _MockModel("D")
        result = run_tts_scaling(model, TINY_IMAGE, QUERY, OPTIONS)
        assert result["answer"] == "D"
