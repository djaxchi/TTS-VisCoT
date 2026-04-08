"""Tests for TTS evaluation helpers (make_predict_fn, evaluate_one, compute_summary)."""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.eval.tts_eval import _find_answer_tag_end, compute_summary, evaluate_one, make_predict_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(100, 100, 100))


def _sample_choices() -> Dict[str, str]:
    return {"A": "Left", "B": "Right", "C": "Front", "D": "Back"}


# ---------------------------------------------------------------------------
# TestMakePredictFn
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestFindAnswerTagEnd  (GRIT answer-token position fix)
# ---------------------------------------------------------------------------


class TestFindAnswerTagEnd:
    def test_returns_prefix_up_to_and_including_answer_tag(self) -> None:
        raw = "<think>some reasoning</think><rethink>more</rethink><answer>B</answer>"
        prefix = _find_answer_tag_end(raw)
        assert prefix is not None
        assert prefix.endswith("<answer>")
        assert "<think>" in prefix

    def test_returns_none_when_no_answer_tag(self) -> None:
        assert _find_answer_tag_end("just some text") is None
        assert _find_answer_tag_end("") is None

    def test_prefix_does_not_include_answer_content(self) -> None:
        raw = "<think>abc</think><answer>C</answer>"
        prefix = _find_answer_tag_end(raw)
        assert prefix is not None
        # The letter C (answer content) must NOT be in the prefix
        assert not prefix.endswith("C")
        assert not prefix.endswith("C<")

    def test_handles_answer_tag_at_start(self) -> None:
        raw = "<answer>A</answer>"
        prefix = _find_answer_tag_end(raw)
        assert prefix == "<answer>"

    def test_handles_multiline_raw_output(self) -> None:
        raw = "<think>\nline1\nline2\n</think>\n<answer>\nA\n</answer>"
        prefix = _find_answer_tag_end(raw)
        assert prefix is not None
        assert prefix.endswith("<answer>")


# ---------------------------------------------------------------------------
# TestEvaluateOneCandidateCorrect
# ---------------------------------------------------------------------------


class TestEvaluateOneCandidateCorrect:
    def test_candidates_have_is_correct_flag(self) -> None:
        outputs = iter(["A", "B", "C", "A", "A", "B", "C", "A", "A"])
        result = evaluate_one(
            image=Image.new("RGB", (32, 32)),
            question="Which?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            correct_answer="A",
            predict_fn=lambda img, p: next(outputs),
            mode="tts",
        )
        for c in result["tts"]["candidates"]:
            assert "is_correct" in c
            assert isinstance(c["is_correct"], bool)

    def test_candidate_is_correct_true_when_normalized_answer_matches(self) -> None:
        result = evaluate_one(
            image=Image.new("RGB", (32, 32)),
            question="Which?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            correct_answer="B",
            predict_fn=lambda img, p: "B",
            mode="tts",
        )
        for c in result["tts"]["candidates"]:
            assert c["is_correct"] is True

    def test_candidate_is_correct_false_when_wrong(self) -> None:
        result = evaluate_one(
            image=Image.new("RGB", (32, 32)),
            question="Which?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            correct_answer="A",
            predict_fn=lambda img, p: "B",
            mode="tts",
        )
        for c in result["tts"]["candidates"]:
            assert c["is_correct"] is False

    def test_candidate_is_correct_none_when_parse_failed(self) -> None:
        result = evaluate_one(
            image=Image.new("RGB", (32, 32)),
            question="Which?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            correct_answer="A",
            predict_fn=lambda img, p: "maybe I think so",
            mode="tts",
        )
        for c in result["tts"]["candidates"]:
            assert c["is_correct"] is None


class TestMakePredictFn:
    def test_make_predict_fn_calls_model_predict_and_extracts_answer(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = {"answer": "B", "bbox_raw": None, "coords": []}

        predict_fn = make_predict_fn(mock_model, temperature=0.0, max_new_tokens=64)
        image = _tiny_image()
        result = predict_fn(image, "Which direction?")

        assert result == "B"
        mock_model.predict.assert_called_once_with(
            image, "Which direction?", temperature=0.0, max_new_tokens=64
        )

    def test_make_predict_fn_returns_empty_string_when_answer_missing(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = {}

        predict_fn = make_predict_fn(mock_model)
        result = predict_fn(_tiny_image(), "probe")

        assert result == ""

    def test_make_predict_fn_default_kwargs_are_passed(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = {"answer": "C"}

        predict_fn = make_predict_fn(mock_model, temperature=0.3, max_new_tokens=128)
        predict_fn(_tiny_image(), "test")

        mock_model.predict.assert_called_once()
        _, call_kwargs = mock_model.predict.call_args
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_new_tokens"] == 128

    def test_make_predict_fn_return_details_includes_token_metadata(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = {"answer": "B"}
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [10]
        mock_tok.convert_ids_to_tokens.return_value = ["B"]
        mock_model._processor = MagicMock(tokenizer=mock_tok)

        predict_fn = make_predict_fn(
            mock_model,
            temperature=0.0,
            max_new_tokens=64,
            return_details=True,
            token_storage_mode="options_only",
        )
        result = predict_fn(_tiny_image(), "Which direction?")

        assert isinstance(result, dict)
        assert result["answer"] == "B"
        assert result["token_metadata"]["storage_mode"] == "options_only"
        assert result["token_metadata"]["generated_token_ids"] == [10]


# ---------------------------------------------------------------------------
# TestOpenEndedEvaluateOne
# ---------------------------------------------------------------------------


class TestOpenEndedEvaluateOne:
    def test_open_ended_baseline_is_correct_when_normalized_matches(self) -> None:
        result = evaluate_one(
            image=Image.new("RGB", (32, 32)),
            question="What is this bird?",
            choices={},
            correct_answer="parrot",
            predict_fn=lambda _img, _p: "A parrot.",
            mode="baseline",
        )
        assert result["baseline"]["is_correct"] is True

    def test_open_ended_baseline_is_correct_false_when_wrong(self) -> None:
        result = evaluate_one(
            image=Image.new("RGB", (32, 32)),
            question="What is this bird?",
            choices={},
            correct_answer="parrot",
            predict_fn=lambda _img, _p: "eagle",
            mode="baseline",
        )
        assert result["baseline"]["is_correct"] is False

    def test_open_ended_tts_correct_with_majority_vote(self) -> None:
        answers = iter(["parrot", "parrot", "bird", "parrot", "parrot", "parrot", "parrot", "parrot", "parrot"])
        result = evaluate_one(
            image=Image.new("RGB", (32, 32)),
            question="What bird is this?",
            choices={},
            correct_answer="parrot",
            predict_fn=lambda _img, _p: next(answers),
            mode="tts",
            tts_kwargs={"allow_early_stop": False},
        )
        assert result["tts"]["winning_answer"] == "parrot"
        assert result["tts"]["is_correct"] is True

    def test_open_ended_prompt_has_no_abcd_choices(self) -> None:
        seen_prompts: list = []
        def capture_predict(_img: Image.Image, prompt: str) -> str:
            seen_prompts.append(prompt)
            return "parrot"
        evaluate_one(
            image=Image.new("RGB", (32, 32)),
            question="What bird is this?",
            choices={},
            correct_answer="parrot",
            predict_fn=capture_predict,
            mode="baseline",
        )
        assert seen_prompts
        assert "A." not in seen_prompts[0]
        assert "Answer with A, B, C, or D" not in seen_prompts[0]


# ---------------------------------------------------------------------------
# TestEvaluateOne
# ---------------------------------------------------------------------------


class TestEvaluateOne:
    def _const_predict(self, answer: str):
        def predict(_img: Image.Image, _prompt: str) -> str:
            return answer

        return predict

    def test_evaluate_one_both_returns_baseline_and_tts_keys(self) -> None:
        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="B",
            predict_fn=self._const_predict("B"),
            mode="both",
        )
        assert result["baseline"] is not None
        assert result["tts"] is not None

    def test_evaluate_one_baseline_only_returns_none_for_tts(self) -> None:
        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="B",
            predict_fn=self._const_predict("B"),
            mode="baseline",
        )
        assert result["baseline"] is not None
        assert result["tts"] is None

    def test_evaluate_one_tts_only_returns_none_for_baseline(self) -> None:
        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="B",
            predict_fn=self._const_predict("B"),
            mode="tts",
        )
        assert result["baseline"] is None
        assert result["tts"] is not None

    def test_evaluate_one_marks_correct_when_answer_matches(self) -> None:
        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="C",
            predict_fn=self._const_predict("Option C"),
            mode="both",
        )
        assert result["baseline"]["is_correct"] is True
        assert result["tts"]["is_correct"] is True

    def test_evaluate_one_marks_incorrect_when_answer_wrong(self) -> None:
        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="A",
            predict_fn=self._const_predict("B"),
            mode="both",
        )
        assert result["baseline"]["is_correct"] is False
        assert result["tts"]["is_correct"] is False

    def test_evaluate_one_baseline_has_required_keys(self) -> None:
        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="B",
            predict_fn=self._const_predict("B"),
            mode="baseline",
        )
        b = result["baseline"]
        assert "raw_output" in b
        assert "normalized_answer" in b
        assert "is_valid" in b
        assert "is_correct" in b

    def test_evaluate_one_tts_has_required_keys(self) -> None:
        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="B",
            predict_fn=self._const_predict("B"),
            mode="tts",
        )
        t = result["tts"]
        for key in ("winning_answer", "is_correct", "stopped_early", "used_candidates",
                    "candidate_answers", "agreement_rate", "vote_margin", "vote_counts"):
            assert key in t, f"missing key: {key}"

    def test_evaluate_one_tts_passes_tts_kwargs_to_pipeline(self) -> None:
        image = _tiny_image()
        choices = _sample_choices()

        with pytest.MonkeyPatch.context() as mp:
            def fake_run_tts_pipeline(*_args, **kwargs):
                assert kwargs["decoding_settings"]["temperature"] == 0.0
                assert kwargs["decoding_settings"]["max_new_tokens"] == 64
                return {
                    "winning_answer": "B",
                    "stopped_early": True,
                    "used_candidates": 3,
                    "candidate_answers": ["B", "B", "A"],
                    "agreement_rate": 2 / 3,
                    "vote_margin": 1,
                    "vote_counts": {"B": 2, "A": 1},
                    "candidates": [],
                }

            mp.setattr("src.eval.tts_eval.run_tts_pipeline", fake_run_tts_pipeline)
            out = evaluate_one(
                image=image,
                question="Which way?",
                choices=choices,
                correct_answer="B",
                predict_fn=self._const_predict("B"),
                mode="tts",
                tts_kwargs={"decoding_settings": {"temperature": 0.0, "max_new_tokens": 64}},
            )

        assert out["tts"]["winning_answer"] == "B"

    def test_evaluate_one_tts_records_early_stop_when_two_agree(self) -> None:
        # Two B answers out of three → early stop
        outputs = iter(["B", "The answer is B", "(A)"])

        def predict(_img: Image.Image, _prompt: str) -> str:
            return next(outputs)

        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="B",
            predict_fn=predict,
            mode="tts",
        )
        assert result["tts"]["stopped_early"] is True
        assert result["tts"]["used_candidates"] == 3

    def test_evaluate_one_tts_runs_full_recipe_when_no_stage1_agreement(self) -> None:
        outputs = iter(["A", "B", "C", "B", "B", "A", "C", "B", "B"])

        def predict(_img: Image.Image, _prompt: str) -> str:
            return next(outputs)

        result = evaluate_one(
            image=_tiny_image(),
            question="Which way?",
            choices=_sample_choices(),
            correct_answer="B",
            predict_fn=predict,
            mode="tts",
        )
        assert result["tts"]["stopped_early"] is False
        assert result["tts"]["used_candidates"] == 9

    def test_evaluate_one_records_question_and_correct_answer(self) -> None:
        result = evaluate_one(
            image=_tiny_image(),
            question="Which direction is east?",
            choices=_sample_choices(),
            correct_answer="D",
            predict_fn=self._const_predict("A"),
            mode="baseline",
        )
        assert result["question"] == "Which direction is east?"
        assert result["correct_answer"] == "D"


# ---------------------------------------------------------------------------
# TestComputeSummary
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def _make_result(
        self,
        baseline_correct: bool,
        tts_correct: bool,
        stopped_early: bool,
        used_candidates: int,
    ) -> Dict:
        return {
            "question": "stub",
            "correct_answer": "A",
            "baseline": {
                "normalized_answer": "A" if baseline_correct else "B",
                "raw_output": "",
                "is_valid": True,
                "is_correct": baseline_correct,
            },
            "tts": {
                "winning_answer": "A" if tts_correct else "B",
                "is_correct": tts_correct,
                "stopped_early": stopped_early,
                "used_candidates": used_candidates,
                "candidate_answers": [],
                "agreement_rate": 1.0,
                "vote_margin": 1,
                "vote_counts": {},
                "candidates": [],
            },
        }

    def test_compute_summary_counts_n_questions(self) -> None:
        results = [self._make_result(True, True, True, 3) for _ in range(4)]
        summary = compute_summary(results)
        assert summary["n_questions"] == 4

    def test_compute_summary_baseline_accuracy(self) -> None:
        results = [
            self._make_result(True, True, True, 3),
            self._make_result(False, True, True, 3),
            self._make_result(True, False, False, 5),
            self._make_result(False, False, False, 5),
        ]
        summary = compute_summary(results)
        assert summary["baseline_accuracy"] == 0.5
        assert summary["baseline_correct"] == 2
        assert summary["baseline_n"] == 4

    def test_compute_summary_tts_accuracy(self) -> None:
        results = [
            self._make_result(True, True, True, 3),
            self._make_result(True, False, True, 3),
            self._make_result(True, True, False, 5),
            self._make_result(True, False, False, 5),
        ]
        summary = compute_summary(results)
        assert summary["tts_accuracy"] == 0.5
        assert summary["tts_correct"] == 2

    def test_compute_summary_tts_early_stop_rate(self) -> None:
        results = [
            self._make_result(True, True, True, 3),
            self._make_result(True, True, True, 3),
            self._make_result(True, True, False, 5),
        ]
        summary = compute_summary(results)
        assert abs(summary["tts_early_stop_rate"] - 2 / 3) < 1e-9

    def test_compute_summary_tts_avg_candidates(self) -> None:
        results = [
            self._make_result(True, True, True, 3),
            self._make_result(True, True, False, 5),
        ]
        summary = compute_summary(results)
        assert summary["tts_avg_candidates"] == 4.0

    def test_compute_summary_handles_empty_results(self) -> None:
        summary = compute_summary([])
        assert summary["n_questions"] == 0
        assert "baseline_accuracy" not in summary

    def test_compute_summary_skips_none_baseline(self) -> None:
        # Simulate a tts-only run
        results = [
            {
                "question": "stub",
                "correct_answer": "A",
                "baseline": None,
                "tts": {
                    "winning_answer": "A",
                    "is_correct": True,
                    "stopped_early": True,
                    "used_candidates": 3,
                    "candidate_answers": [],
                    "agreement_rate": 1.0,
                    "vote_margin": 1,
                    "vote_counts": {},
                    "candidates": [],
                },
            }
        ]
        summary = compute_summary(results)
        assert "baseline_accuracy" not in summary
        assert summary["tts_accuracy"] == 1.0
