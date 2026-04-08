"""Tests for candidate view artifact saving and paraphrase caching in experiments.run_tts_eval."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from experiments.run_tts_eval import (
    _paraphrase_is_usable,
    get_or_generate_model_paraphrase,
    load_paraphrase_cache,
    load_static_paraphrase_cache,
    resolve_static_paraphrase,
    save_candidate_views,
)


def _choices() -> dict[str, str]:
    return {"A": "left", "B": "right", "C": "front", "D": "back"}


def test_save_candidate_views_writes_images_and_prompts(tmp_path: Path) -> None:
    rows = save_candidate_views(
        save_dir=tmp_path,
        image_id="sample_1",
        image=Image.new("RGB", (32, 32), color=(120, 80, 40)),
        question="Where is the sign?",
        choices=_choices(),
        image_preset="strong",
    )

    assert len(rows) == 9
    assert (tmp_path / "candidate_views" / "sample_1" / "candidate_1.png").exists()
    assert (tmp_path / "candidate_views" / "sample_1" / "candidate_9.png").exists()
    assert (tmp_path / "candidate_views" / "sample_1" / "candidate_2_prompt.txt").exists()


def test_save_candidate_views_rows_include_transform_and_text_ids(tmp_path: Path) -> None:
    rows = save_candidate_views(
        save_dir=tmp_path,
        image_id="sample_2",
        image=Image.new("RGB", (32, 32), color=(10, 20, 30)),
        question="Which option is correct?",
        choices=_choices(),
        image_preset="strong",
    )

    row = rows[2]
    assert row["sample_id"] == "sample_2"
    assert "image_transform_id" in row
    assert "text_variant_id" in row
    assert row["prompt"].strip() != ""


def test_load_paraphrase_cache_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_paraphrase_cache(tmp_path / "missing.jsonl") == {}


def test_load_static_paraphrase_cache_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_static_paraphrase_cache(tmp_path / "missing_static.jsonl") == {}


def test_resolve_static_paraphrase_prefers_sample_id_then_question() -> None:
    static_cache = {
        "sample_1": {
            "sample_id": "sample_1",
            "question": "Where is the sign?",
            "model_paraphrase": "Where is the sign located relative to the viewer?",
        },
        "q::Which direction is left?": {
            "sample_id": "",
            "question": "Which direction is left?",
            "model_paraphrase": "What is the leftward direction?",
        },
    }

    by_id = resolve_static_paraphrase(static_cache, sample_id="sample_1", question="Where is the sign?")
    by_question = resolve_static_paraphrase(
        static_cache,
        sample_id="unknown",
        question="Which direction is left?",
    )
    missing = resolve_static_paraphrase(static_cache, sample_id="unknown", question="No match")

    assert by_id == "Where is the sign located relative to the viewer?"
    assert by_question == "What is the leftward direction?"
    assert missing is None


def test_get_or_generate_model_paraphrase_uses_cached_value_without_model_call(tmp_path: Path) -> None:
    cache_path = tmp_path / "paraphrases.jsonl"
    cache = {
        "sample_1": {
            "cache_version": 2,
            "sample_id": "sample_1",
            "question": "Where is the sign?",
            "model_paraphrase": "Where does the sign appear relative to the viewer?",
        }
    }
    model = MagicMock()

    out = get_or_generate_model_paraphrase(
        cache=cache,
        cache_path=cache_path,
        sample_id="sample_1",
        image=Image.new("RGB", (32, 32), color=(1, 2, 3)),
        question="Where is the sign?",
        choices=_choices(),
        model=model,
        model_label="baseline-model",
    )

    assert out == "Where does the sign appear relative to the viewer?"
    model.predict.assert_not_called()


def test_get_or_generate_model_paraphrase_generates_and_persists_on_cache_miss(tmp_path: Path) -> None:
    cache_path = tmp_path / "paraphrases.jsonl"
    cache: dict[str, dict[str, str]] = {}
    model = MagicMock()
    model.predict.return_value = {"answer": "Rephrase the question as: Which direction is the sign located?"}

    out = get_or_generate_model_paraphrase(
        cache=cache,
        cache_path=cache_path,
        sample_id="sample_2",
        image=Image.new("RGB", (32, 32), color=(1, 2, 3)),
        question="Where is the sign?",
        choices=_choices(),
        model=model,
        model_label="baseline-model",
    )

    assert out != ""
    assert "sample_2" in cache
    assert cache["sample_2"]["model_paraphrase"] == out
    assert cache["sample_2"]["attempts"][0]["accepted"] is True
    saved = cache_path.read_text(encoding="utf-8")
    assert "sample_2" in saved


def test_paraphrase_is_usable_rejects_known_template_prefixes() -> None:
    assert _paraphrase_is_usable(
        "Inspect the image carefully and choose the best option. Where is the sign?",
        "Where is the sign?",
    ) is False


def test_get_or_generate_model_paraphrase_regenerates_stale_weak_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "paraphrases.jsonl"
    cache = {
        "sample_3": {
            "cache_version": 1,
            "sample_id": "sample_3",
            "question": "Where is the sign?",
            "model_paraphrase": "Inspect the image carefully and choose the best option. Where is the sign?",
        }
    }
    model = MagicMock()
    model.predict.side_effect = [
        {"answer": "Where is the sign situated relative to the viewer?"},
    ]

    out = get_or_generate_model_paraphrase(
        cache=cache,
        cache_path=cache_path,
        sample_id="sample_3",
        image=Image.new("RGB", (32, 32), color=(1, 2, 3)),
        question="Where is the sign?",
        choices=_choices(),
        model=model,
        model_label="baseline-model",
    )

    assert out == "Where is the sign situated relative to the viewer?"
    assert model.predict.call_count == 1


def test_get_or_generate_model_paraphrase_retries_when_first_attempt_is_template_like(tmp_path: Path) -> None:
    cache_path = tmp_path / "paraphrases.jsonl"
    cache: dict[str, dict[str, str]] = {}
    model = MagicMock()
    model.predict.side_effect = [
        {"answer": "Inspect the image carefully and choose the best option. Where is the sign?"},
        {"answer": "Where is the sign positioned relative to the viewer?"},
    ]

    out = get_or_generate_model_paraphrase(
        cache=cache,
        cache_path=cache_path,
        sample_id="sample_4",
        image=Image.new("RGB", (32, 32), color=(1, 2, 3)),
        question="Where is the sign?",
        choices=_choices(),
        model=model,
        model_label="baseline-model",
    )

    assert out == "Where is the sign positioned relative to the viewer?"
    assert model.predict.call_count == 2
    assert cache["sample_4"]["attempts"][0]["accepted"] is False
    assert cache["sample_4"]["attempts"][1]["accepted"] is True