"""Tests for checkpoint saving and resume in experiments/run_comparison.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.run_model_benchmark import (
    DEFAULT_QUESTION_COUNT,
    InferenceResult,
    build_model_configs,
    load_checkpoint,
    model_is_complete,
    save_checkpoint,
    task_is_complete,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_result(qid: str = "q1", correct: bool = True) -> InferenceResult:
    return InferenceResult(
        question_id=qid,
        question="What colour?",
        answer="red",
        references=["red"],
        correct=correct,
        tokens=12,
        elapsed_s=1.5,
    )


# ---------------------------------------------------------------------------
# TestSaveCheckpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_save_checkpoint_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        save_checkpoint(out, {"ModelA": {"vqa": [_make_result()]}}, ["ModelA"], ["vqa"])
        assert out.exists()

    def test_save_checkpoint_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "cp.json"
        save_checkpoint(out, {"ModelA": {"vqa": [_make_result()]}}, ["ModelA"], ["vqa"])
        assert out.exists()

    def test_save_checkpoint_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        save_checkpoint(out, {"ModelA": {"vqa": [_make_result()]}}, ["ModelA"], ["vqa"])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_save_checkpoint_contains_model_key(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        save_checkpoint(out, {"ModelA": {"vqa": [_make_result()]}}, ["ModelA"], ["vqa"])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "ModelA" in data

    def test_save_checkpoint_contains_task_key(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        save_checkpoint(out, {"ModelA": {"vqa": [_make_result()]}}, ["ModelA"], ["vqa"])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "vqa" in data["ModelA"]

    def test_save_checkpoint_result_fields_present(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        save_checkpoint(out, {"ModelA": {"vqa": [_make_result("q42")]}}, ["ModelA"], ["vqa"])
        data = json.loads(out.read_text(encoding="utf-8"))
        row = data["ModelA"]["vqa"][0]
        for key in ("question_id", "question", "answer", "references", "correct", "tokens", "elapsed_s"):
            assert key in row, f"Missing key: {key}"

    def test_save_checkpoint_result_values_correct(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        r = _make_result("q99", correct=False)
        save_checkpoint(out, {"M": {"ocr": [r]}}, ["M"], ["ocr"])
        data = json.loads(out.read_text(encoding="utf-8"))
        row = data["M"]["ocr"][0]
        assert row["question_id"] == "q99"
        assert row["correct"] is False
        assert row["elapsed_s"] == pytest.approx(1.5)

    def test_save_checkpoint_multiple_models_and_tasks(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        results = {
            "ModelA": {"vqa": [_make_result("a1")], "counting": [_make_result("a2")]},
            "ModelB": {"vqa": [_make_result("b1")]},
        }
        save_checkpoint(out, results, ["ModelA", "ModelB"], ["vqa", "counting"])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["ModelA"]["vqa"]) == 1
        assert len(data["ModelA"]["counting"]) == 1
        assert len(data["ModelB"]["vqa"]) == 1

    def test_save_checkpoint_overwrites_existing_file(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        out.write_text('{"stale": true}', encoding="utf-8")
        save_checkpoint(out, {"ModelA": {"vqa": [_make_result()]}}, ["ModelA"], ["vqa"])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "stale" not in data

    def test_save_checkpoint_empty_results_writes_empty_structure(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        save_checkpoint(out, {}, [], [])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == {}

    def test_save_checkpoint_missing_model_in_results_still_writes(self, tmp_path: Path) -> None:
        """Model listed in model_labels but not yet in all_results (e.g. loading failed)."""
        out = tmp_path / "cp.json"
        save_checkpoint(out, {}, ["ModelA"], ["vqa"])
        data = json.loads(out.read_text(encoding="utf-8"))
        # ModelA should appear with empty task dict
        assert "ModelA" in data
        assert data["ModelA"] == {}


# ---------------------------------------------------------------------------
# TestLoadCheckpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def test_load_checkpoint_returns_empty_dict_when_file_missing(self, tmp_path: Path) -> None:
        result = load_checkpoint(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_checkpoint_returns_parsed_json(self, tmp_path: Path) -> None:
        cp = tmp_path / "cp.json"
        cp.write_text(json.dumps({"ModelA": {"vqa": []}}), encoding="utf-8")
        result = load_checkpoint(cp)
        assert "ModelA" in result

    def test_load_checkpoint_returns_empty_dict_on_invalid_json(self, tmp_path: Path) -> None:
        cp = tmp_path / "cp.json"
        cp.write_text("not json {{", encoding="utf-8")
        result = load_checkpoint(cp)
        assert result == {}


# ---------------------------------------------------------------------------
# TestModelIsComplete
# ---------------------------------------------------------------------------


class TestModelIsComplete:
    def test_model_is_complete_true_when_all_tasks_have_enough_results(self) -> None:
        checkpoint = {
            "ModelA": {
                "vqa": [{"question_id": f"q{i}"} for i in range(5)],
                "counting": [{"question_id": f"q{i}"} for i in range(5)],
            }
        }
        assert model_is_complete("ModelA", checkpoint, ["vqa", "counting"], n=5) is True

    def test_model_is_complete_false_when_task_missing(self) -> None:
        checkpoint = {"ModelA": {"vqa": [{"question_id": "q1"}] * 5}}
        assert model_is_complete("ModelA", checkpoint, ["vqa", "counting"], n=5) is False

    def test_model_is_complete_false_when_task_has_fewer_results_than_n(self) -> None:
        checkpoint = {"ModelA": {"vqa": [{"question_id": "q1"}] * 3}}
        assert model_is_complete("ModelA", checkpoint, ["vqa"], n=5) is False

    def test_model_is_complete_false_when_model_missing_from_checkpoint(self) -> None:
        assert model_is_complete("ModelA", {}, ["vqa"], n=5) is False

    def test_model_is_complete_true_when_n_is_zero(self) -> None:
        assert model_is_complete("ModelA", {}, ["vqa"], n=0) is True


# ---------------------------------------------------------------------------
# TestTaskIsComplete
# ---------------------------------------------------------------------------


class TestTaskIsComplete:
    def _cp(self, label: str, task: str, count: int) -> dict:
        return {label: {task: [{"question_id": f"q{i}"} for i in range(count)]}}

    def test_complete_when_enough_results(self) -> None:
        cp = self._cp("M", "vqa", 5)
        assert task_is_complete("M", "vqa", cp, n=5) is True

    def test_incomplete_when_too_few_results(self) -> None:
        cp = self._cp("M", "vqa", 3)
        assert task_is_complete("M", "vqa", cp, n=5) is False

    def test_incomplete_when_task_missing(self) -> None:
        assert task_is_complete("M", "vqa", {}, n=5) is False

    def test_incomplete_when_model_missing(self) -> None:
        assert task_is_complete("ModelX", "vqa", {"Other": {}}, n=5) is False

    def test_redo_task_forces_incomplete_even_if_checkpoint_has_results(self) -> None:
        cp = self._cp("M", "counting", 5)
        assert task_is_complete("M", "counting", cp, n=5, redo_tasks={"counting"}) is False

    def test_redo_task_does_not_affect_other_tasks(self) -> None:
        cp = self._cp("M", "vqa", 5)
        # counting is the redo task but vqa should still be reported complete
        assert task_is_complete("M", "vqa", cp, n=5, redo_tasks={"counting"}) is True

    def test_complete_when_n_is_zero(self) -> None:
        assert task_is_complete("M", "vqa", {}, n=0) is True

    @pytest.mark.parametrize("task", ["vqa", "counting", "ocr"])
    def test_redo_any_task_forces_that_task_incomplete(self, task: str) -> None:
        cp = {"M": {task: [{"question_id": "q1"}] * 10}}
        assert task_is_complete("M", task, cp, n=5, redo_tasks={task}) is False


# ---------------------------------------------------------------------------
# TestRunConfiguration
# ---------------------------------------------------------------------------


class TestRunConfiguration:
    def test_default_question_count_is_fifty(self) -> None:
        assert DEFAULT_QUESTION_COUNT == 50

    def test_build_model_configs_excludes_deepeyes_when_requested(self) -> None:
        labels = [
            cfg["label"]
            for cfg in build_model_configs(
                include_viscot=True,
                include_deepeyes=False,
                load_in_8bit=True,
            )
        ]

        assert "DeepEyesV2-RL (7B)" not in labels

    def test_build_model_configs_keeps_fast_models_in_expected_order(self) -> None:
        labels = [
            cfg["label"]
            for cfg in build_model_configs(
                include_viscot=True,
                include_deepeyes=False,
                load_in_8bit=True,
            )
        ]

        assert labels == [
            "VisCoT (7B)",
            "Qwen2.5-VL (3B, no CoT)",
            "GRIT (3B)",
        ]


# ---------------------------------------------------------------------------
# TestRunModelOnSamplesErrorHandling
# ---------------------------------------------------------------------------


class TestRunModelOnSamplesErrorHandling:
    """Verify that a crash on one sample does not abort the rest."""

    def _make_sample(self, qid: str) -> dict:
        from PIL import Image as PILImage
        return {
            "question_id": qid,
            "question": "What colour?",
            "answer": "red",
            "image": PILImage.new("RGB", (64, 64), color=(255, 0, 0)),
        }

    def test_failed_sample_is_skipped_and_run_continues(self) -> None:
        from experiments.run_model_benchmark import _run_model_on_samples

        model = MagicMock()
        # First call raises, second succeeds
        model.predict.side_effect = [RuntimeError("GPU OOM"), MagicMock(get=lambda k, d="": "red")]

        samples = [self._make_sample("q1"), self._make_sample("q2")]
        with patch("src.eval.vqa_eval.evaluate_vqa", return_value=True):
            results = _run_model_on_samples(model, samples, 0.0, 256, "TestModel", "vqa")

        # Only the successful sample produces a result
        assert len(results) == 1
        assert results[0].question_id == "q2"

    def test_all_samples_fail_returns_empty_list(self) -> None:
        from experiments.run_model_benchmark import _run_model_on_samples

        model = MagicMock()
        model.predict.side_effect = RuntimeError("always fails")
        samples = [self._make_sample("q1"), self._make_sample("q2")]
        with patch("src.eval.vqa_eval.evaluate_vqa", return_value=False):
            results = _run_model_on_samples(model, samples, 0.0, 256, "TestModel", "vqa")
        assert results == []


class TestRunModelOnSamplesTTS:
    def _make_sample(self, qid: str) -> dict:
        from PIL import Image as PILImage

        return {
            "question_id": qid,
            "question": "What is shown?",
            "answer": "red car",
            "image": PILImage.new("RGB", (64, 64), color=(255, 0, 0)),
        }

    def test_tts_stores_all_candidate_answers_and_replays_3_vs_5_votes(self) -> None:
        from experiments.run_model_benchmark import _run_model_on_samples

        model = MagicMock()
        model.predict.side_effect = [
            {"answer": "red car"},
            {"answer": "a red car"},
            {"answer": "blue car"},
            {"answer": "blue car"},
            {"answer": "blue car"},
        ]
        samples = [self._make_sample("q1")]

        results = _run_model_on_samples(
            model,
            samples,
            temperature=0.0,
            max_new_tokens=256,
            model_label="Qwen2.5-VL (3B, no CoT)",
            task="vqa",
            tts_candidates=5,
        )

        assert len(results) == 1
        row = results[0]
        assert row.candidate_answers == [
            "red car",
            "a red car",
            "blue car",
            "blue car",
            "blue car",
        ]
        assert row.candidate_answers_normalized == [
            "red car",
            "red car",
            "blue car",
            "blue car",
            "blue car",
        ]
        assert row.candidate_answer_token_ids is not None
        assert row.candidate_answer_tokens is not None
        assert len(row.candidate_answer_token_ids) == 5
        assert len(row.candidate_answer_tokens) == 5

        v3 = row.voting.get("majority_3", {})
        v5 = row.voting.get("majority_5", {})
        assert v3.get("answer") == "red car"
        assert v5.get("answer") == "blue car"

    def test_save_checkpoint_persists_tts_trace_fields(self, tmp_path: Path) -> None:
        out = tmp_path / "cp.json"
        r = InferenceResult(
            question_id="q1",
            question="What is shown?",
            answer="blue car",
            references=["red car"],
            correct=False,
            tokens=50,
            elapsed_s=2.5,
            candidate_answers=["red car", "a red car", "blue car", "blue car", "blue car"],
            candidate_answers_normalized=["red car", "red car", "blue car", "blue car", "blue car"],
            voting={
                "majority_3": {"answer": "red car", "correct": True},
                "majority_5": {"answer": "blue car", "correct": False},
            },
        )
        save_checkpoint(out, {"ModelA": {"vqa": [r]}}, ["ModelA"], ["vqa"])

        data = json.loads(out.read_text(encoding="utf-8"))
        row = data["ModelA"]["vqa"][0]
        assert "candidate_answers" in row
        assert "candidate_answers_normalized" in row
        assert "candidate_answer_token_ids" in row
        assert "candidate_answer_tokens" in row
        assert "voting" in row
        assert row["voting"]["majority_3"]["answer"] == "red car"

    def test_tts9_uses_majority_9_and_records_vote(self) -> None:
        from experiments.run_model_benchmark import _run_model_on_samples

        model = MagicMock()
        model.predict.side_effect = [
            {"answer": "cat"},
            {"answer": "dog"},
            {"answer": "dog"},
            {"answer": "dog"},
            {"answer": "dog"},
            {"answer": "dog"},
            {"answer": "dog"},
            {"answer": "cat"},
            {"answer": "cat"},
        ]
        samples = [self._make_sample("q1")]

        results = _run_model_on_samples(
            model,
            samples,
            temperature=0.0,
            max_new_tokens=256,
            model_label="Qwen2.5-VL (3B, no CoT)",
            task="vqa",
            tts_candidates=9,
        )

        assert len(results) == 1
        row = results[0]
        assert row.voting is not None
        assert row.voting["majority_9"]["answer"] == "dog"
        assert row.answer == "dog"


class TestBuildTTSQueries:
    def test_paraphrase_injected_at_candidate_4_and_9(self) -> None:
        from experiments.run_model_benchmark import _build_tts_queries

        question = "What is shown?"
        para = "Which object is visible?"
        queries = _build_tts_queries(
            question,
            9,
            paraphrase=para,
            paraphrase_slots_1based=[4, 9],
        )

        assert len(queries) == 9
        assert queries[3] == para
        assert queries[8] == para
        assert queries[0] != para


class TestTokenizationForStorage:
    def test_tokenize_text_for_storage_uses_tokenizer_when_available(self) -> None:
        from experiments.run_model_benchmark import _tokenize_text_for_storage

        class _Tok:
            def encode(self, text: str, add_special_tokens: bool = False):
                assert add_special_tokens is False
                return [11, 22]

            def convert_ids_to_tokens(self, ids):
                return [f"tok_{i}" for i in ids]

        class _Proc:
            tokenizer = _Tok()

        class _Model:
            _processor = _Proc()

        ids, toks = _tokenize_text_for_storage(_Model(), "hello world")
        assert ids == [11, 22]
        assert toks == ["tok_11", "tok_22"]

    def test_tokenize_text_for_storage_falls_back_to_whitespace_tokens(self) -> None:
        from experiments.run_model_benchmark import _tokenize_text_for_storage

        class _Model:
            _processor = None
            _tokenizer = None

        ids, toks = _tokenize_text_for_storage(_Model(), "red car")
        assert ids == []
        assert toks == ["red", "car"]
