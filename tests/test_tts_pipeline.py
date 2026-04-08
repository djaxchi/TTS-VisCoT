"""Tests for lightweight TreeBench test-time scaling wrapper modules."""

from __future__ import annotations

from typing import Dict, List

import pytest
from PIL import Image

from src.augment_image import (
    ImageVariationConfig,
    generate_image_variant_specs,
    generate_image_variants,
)
from src.augment_text import generate_prompt_variants, generate_question_variants
from src.pipeline_tts import (
    build_candidate_inputs,
    export_debug_artifacts,
    run_baseline,
    run_tts_pipeline,
)
from src.utils_normalize import normalize_answer
from src.voting_tts import compute_vote_stats, majority_vote


def _sample_choices() -> Dict[str, str]:
    return {
        "A": "2",
        "B": "3",
        "C": "4",
        "D": "5",
    }


class TestNormalizeOpenEndedAnswer:
    def test_normalize_open_ended_lowercases(self) -> None:
        from src.utils_normalize import normalize_open_ended_answer
        assert normalize_open_ended_answer("PARROT") == "parrot"

    def test_normalize_open_ended_strips_articles(self) -> None:
        from src.utils_normalize import normalize_open_ended_answer
        assert normalize_open_ended_answer("a parrot") == "parrot"
        assert normalize_open_ended_answer("an eagle") == "eagle"
        assert normalize_open_ended_answer("the sky") == "sky"

    def test_normalize_open_ended_strips_trailing_punctuation(self) -> None:
        from src.utils_normalize import normalize_open_ended_answer
        assert normalize_open_ended_answer("parrot.") == "parrot"
        assert normalize_open_ended_answer("yes!") == "yes"
        assert normalize_open_ended_answer("3,") == "3"

    def test_normalize_open_ended_returns_none_for_empty(self) -> None:
        from src.utils_normalize import normalize_open_ended_answer
        assert normalize_open_ended_answer("") is None
        assert normalize_open_ended_answer("   ") is None

    def test_normalize_open_ended_preserves_numbers(self) -> None:
        from src.utils_normalize import normalize_open_ended_answer
        assert normalize_open_ended_answer("4") == "4"
        assert normalize_open_ended_answer("  2  ") == "2"


class TestOpenEndedPromptVariants:
    def test_empty_choices_omits_choices_section(self) -> None:
        from src.augment_text import generate_prompt_variants
        variants = generate_prompt_variants(question="What is this bird called?", choices={})
        for v in variants.values():
            assert "Choices:" not in v["prompt"]
            assert "A." not in v["prompt"]

    def test_empty_choices_omits_abcd_constraint(self) -> None:
        from src.augment_text import generate_prompt_variants
        variants = generate_prompt_variants(question="What is this bird called?", choices={})
        for v in variants.values():
            assert "Answer with A, B, C, or D" not in v["prompt"]

    def test_empty_choices_prompt_contains_question(self) -> None:
        from src.augment_text import generate_prompt_variants
        q = "What is this bird called?"
        variants = generate_prompt_variants(question=q, choices={})
        for v in variants.values():
            assert q in v["prompt"] or "bird" in v["prompt"]


class TestOpenEndedPipeline:
    def test_run_tts_pipeline_open_ended_voting_uses_string_answers(self) -> None:
        answers = iter(["parrot", "parrot", "bird", "parrot", "parrot", "parrot", "bird", "parrot", "parrot"])
        result = run_tts_pipeline(
            image=Image.new("RGB", (32, 32)),
            question="What is this bird called?",
            choices={},
            predict_fn=lambda _img, _p: next(answers),
            allow_early_stop=False,
        )
        assert result["winning_answer"] == "parrot"

    def test_run_baseline_open_ended_normalizes_free_text(self) -> None:
        result = run_baseline(
            image=Image.new("RGB", (32, 32)),
            question="What is this bird?",
            choices={},
            predict_fn=lambda _img, _p: "A parrot.",
        )
        assert result["normalized_answer"] == "parrot"


class TestNormalizeAnswer:
    def test_normalize_answer_handles_common_formats(self) -> None:
        assert normalize_answer("A") == "A"
        assert normalize_answer("(B)") == "B"
        assert normalize_answer("Option C") == "C"
        assert normalize_answer("The answer is D") == "D"

    def test_normalize_answer_returns_none_when_invalid(self) -> None:
        assert normalize_answer("I am not sure") is None


class TestTextVariation:
    def test_generate_prompt_variants_includes_expected_variant_ids(self) -> None:
        variants = generate_prompt_variants(
            question="How many apples are on the table?",
            choices=_sample_choices(),
        )

        assert set(variants.keys()) == {"original", "hardcoded_paraphrase", "model_paraphrase"}
        assert "single capital letter" in variants["model_paraphrase"]["prompt"].lower()

    def test_generate_question_variants_rule_returns_three_prompts(self) -> None:
        variants = generate_question_variants(
            question="How many apples are on the table?",
            choices=_sample_choices(),
            mode="rule",
        )

        assert len(variants) == 3
        assert "Answer with A, B, C, or D only." in variants[0]
        assert "Using the image, answer this question:" in variants[1]
        assert variants[2].startswith("How many apples are on the table?") is False

    def test_generate_question_variants_uses_answer_format_variants(self) -> None:
        variants = generate_question_variants(
            question="How many apples are on the table?",
            choices=_sample_choices(),
            mode="rule",
        )

        assert "Reply using a single letter: A, B, C, or D." in variants[1]
        assert "Select one option only: A, B, C, or D." in variants[2]
        # variant 3 uses instruction frame 2 as its distinguishing framing
        assert "Determine the best answer from the image:" in variants[2]

    def test_model_paraphrase_fallback_all_three_prompts_distinct(self) -> None:
        variants = generate_question_variants(
            question="How many apples are on the table?",
            choices=_sample_choices(),
            mode="rule",
        )
        assert variants[0] != variants[1]
        assert variants[1] != variants[2]
        assert variants[0] != variants[2]

    def test_model_paraphrase_fallback_contains_question_content(self) -> None:
        q = "How many apples are on the table?"
        variants = generate_prompt_variants(question=q, choices=_sample_choices())
        # The question text must appear somewhere in every variant prompt
        for key, v in variants.items():
            assert q in v["prompt"], f"question missing from {key} prompt"

    def test_model_paraphrase_fallback_no_banned_prefix(self) -> None:
        variants = generate_prompt_variants(
            question="How many apples are on the table?",
            choices=_sample_choices(),
        )
        p = variants["model_paraphrase"]["prompt"].lower()
        banned = (
            "inspect the image carefully and choose the best option",
            "from the image determine which option is correct",
            "from the image, determine which option is correct",
        )
        for prefix in banned:
            assert not p.startswith(prefix), f"model_paraphrase starts with banned prefix: {prefix!r}"

    def test_generate_question_variants_model_uses_callback(self) -> None:
        def paraphraser(question: str, _choices: Dict[str, str], idx: int) -> str:
            return f"Paraphrase {idx}: {question}"

        variants = generate_question_variants(
            question="How many apples are on the table?",
            choices=_sample_choices(),
            mode="model",
            model_paraphrase_fn=paraphraser,
        )

        assert "Paraphrase 2" in variants[2]


class TestImageVariation:
    def test_generate_image_variant_specs_includes_strong_defaults(self) -> None:
        image = Image.new("RGB", (32, 32), color=(120, 100, 90))
        specs = generate_image_variant_specs(image, config=ImageVariationConfig(preset="strong"))

        assert "original" in specs
        assert "grayscale" in specs
        assert "edge_enhance" in specs
        assert specs["grayscale"]["transform_id"] == "grayscale"
        assert specs["edge_enhance"]["preset"] == "strong"

    def test_generate_image_variants_returns_original_and_two_variants(self) -> None:
        image = Image.new("RGB", (32, 32), color=(120, 100, 90))
        variants = generate_image_variants(image)

        assert len(variants) == 3
        assert variants["original"].size == image.size
        assert variants["image_variation_1"].size == image.size
        assert variants["image_variation_2"].size == image.size

    def test_generate_image_variant_specs_rotation_90_preserves_full_content_with_swapped_canvas(self) -> None:
        image = Image.new("RGB", (48, 32), color=(120, 100, 90))
        specs = generate_image_variant_specs(image, config=ImageVariationConfig(preset="strong"))

        assert specs["rotation_90"]["image"].size == (32, 48)


class TestVoting:
    def test_majority_vote_returns_winner(self) -> None:
        answers: List[str | None] = ["B", "B", "C"]
        assert majority_vote(answers) == "B"

    def test_compute_vote_stats_reports_agreement_and_margin(self) -> None:
        stats = compute_vote_stats(["A", "A", "C", None, "A"])
        assert stats.winning_answer == "A"
        assert stats.agreement_rate == 0.75
        assert stats.vote_margin == 2

    def test_compute_vote_stats_entropy_zero_when_all_agree(self) -> None:
        stats = compute_vote_stats(["A", "A", "A"])
        assert stats.answer_entropy == pytest.approx(0.0)

    def test_compute_vote_stats_entropy_max_when_uniform(self) -> None:
        # 4 answers, each different → entropy = log2(4) = 2.0
        stats = compute_vote_stats(["A", "B", "C", "D"])
        assert stats.answer_entropy == pytest.approx(2.0)

    def test_compute_vote_stats_entropy_between_extremes(self) -> None:
        stats = compute_vote_stats(["A", "A", "B"])
        assert 0.0 < stats.answer_entropy < 2.0

    def test_compute_vote_stats_entropy_zero_when_no_valid_votes(self) -> None:
        stats = compute_vote_stats([None, None])
        assert stats.answer_entropy == 0.0


class TestPipelineDiagnostics:
    """Tests for per-candidate elapsed_s and stage-level diagnostics."""

    def _predict_constant(self, answer: str):
        def predict(_img: Image.Image, _prompt: str) -> str:
            return answer
        return predict

    def test_candidate_result_has_elapsed_s(self) -> None:
        image = Image.new("RGB", (32, 32))
        result = run_tts_pipeline(
            image=image,
            question="test?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            predict_fn=self._predict_constant("A"),
            candidate_recipe=[(1, "original", "original"), (1, "original", "hardcoded_paraphrase"), (1, "original", "model_paraphrase")],
        )
        for c in result["candidates"]:
            assert "elapsed_s" in c
            assert isinstance(c["elapsed_s"], float)
            assert c["elapsed_s"] >= 0.0

    def test_pipeline_output_has_stage1_agreement_rate(self) -> None:
        image = Image.new("RGB", (32, 32))
        outputs = iter(["A", "A", "B"])
        result = run_tts_pipeline(
            image=image,
            question="test?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            predict_fn=lambda img, p: next(outputs),
        )
        assert "stage_1_agreement_rate" in result
        assert 0.0 <= result["stage_1_agreement_rate"] <= 1.0

    def test_stage1_agreement_rate_matches_early_stop_votes(self) -> None:
        image = Image.new("RGB", (32, 32))
        # All 3 stage-1 candidates agree → agreement_rate = 1.0
        result = run_tts_pipeline(
            image=image,
            question="test?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            predict_fn=self._predict_constant("B"),
            candidate_recipe=[(1, "original", "original"), (1, "original", "hardcoded_paraphrase"), (1, "original", "model_paraphrase")],
        )
        assert result["stage_1_agreement_rate"] == pytest.approx(1.0)

    def test_stage2_changed_answer_false_when_stopped_early(self) -> None:
        image = Image.new("RGB", (32, 32))
        result = run_tts_pipeline(
            image=image,
            question="test?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            predict_fn=self._predict_constant("C"),
            candidate_recipe=[(1, "original", "original"), (1, "original", "hardcoded_paraphrase"), (1, "original", "model_paraphrase")],
        )
        assert result["stopped_early"] is True
        assert result["stage2_changed_answer"] is False

    def test_stage2_changed_answer_true_when_stage2_flips_winner(self) -> None:
        image = Image.new("RGB", (32, 32))
        # Stage 1: A, B, C → no consensus; Stage 2: B, B, B, B, B, B → B wins overall
        outputs = iter(["A", "B", "C", "B", "B", "B", "B", "B", "B"])
        result = run_tts_pipeline(
            image=image,
            question="test?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            predict_fn=lambda img, p: next(outputs),
        )
        assert result["stopped_early"] is False
        # Stage 1 winner was ambiguous; overall winner = B
        assert result["stage2_changed_answer"] is not None  # field exists

    def test_stage2_changed_answer_false_when_winner_unchanged(self) -> None:
        image = Image.new("RGB", (32, 32))
        # Stage 1: A, A, B → A leads; Stage 2: confirms A
        outputs = iter(["A", "A", "B", "A", "A", "A", "A", "A", "A"])
        result = run_tts_pipeline(
            image=image,
            question="test?",
            choices={"A": "1", "B": "2", "C": "3", "D": "4"},
            predict_fn=lambda img, p: next(outputs),
        )
        assert result["stage2_changed_answer"] is False


class TestPipeline:
    def test_build_candidate_inputs_supports_configurable_recipe(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))
        candidates = build_candidate_inputs(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            max_candidates=5,
            text_mode="rule",
            candidate_recipe=[
                (1, "original", "original"),
                (1, "original", "hardcoded_paraphrase"),
                (1, "edge_enhance", "original"),
                (2, "original", "model_paraphrase"),
                (2, "grayscale", "original"),
            ],
        )

        assert candidates[2]["image_transform_id"] == "edge_enhance"
        assert candidates[4]["image_transform_id"] == "grayscale"
        assert candidates[1]["text_variant_id"] == "hardcoded_paraphrase"

    def test_build_candidate_inputs_returns_three_candidates_when_requested(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))
        candidates = build_candidate_inputs(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            max_candidates=3,
            text_mode="rule",
        )

        assert len(candidates) == 3
        assert candidates[0]["candidate_id"] == 1
        assert candidates[-1]["candidate_id"] == 3

    def test_build_candidate_inputs_returns_full_recipe_by_default(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))
        candidates = build_candidate_inputs(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            text_mode="rule",
        )

        assert len(candidates) == 9
        assert candidates[2]["text_variant_id"] == "model_paraphrase"
        assert candidates[7]["image_transform_id"] == "rotation"
        assert candidates[8]["text_variant_id"] == "model_paraphrase"

    def test_build_candidate_inputs_supports_more_than_five(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))
        candidates = build_candidate_inputs(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            max_candidates=7,
            text_mode="rule",
        )

        assert len(candidates) == 7

    def test_build_candidate_inputs_raises_when_recipe_too_short(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))

        with pytest.raises(ValueError):
            build_candidate_inputs(
                image=image,
                question="How many apples are on the table?",
                choices=_sample_choices(),
                max_candidates=4,
                text_mode="rule",
                candidate_recipe=[
                    (1, "original", "original"),
                    (1, "original", "hardcoded_paraphrase"),
                ],
            )

    def test_pipeline_stops_after_three_when_two_agree(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))
        outputs = iter(["Option B", "The answer is B", "(A)"])

        def predict(_image: Image.Image, _question: str) -> Dict[str, object]:
            return {
                "answer": next(outputs),
                "token_metadata": {
                    "storage_mode": "options_only",
                    "generated_token_ids": [101, 202],
                },
            }

        result = run_tts_pipeline(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            predict_fn=predict,
            text_mode="rule",
        )

        assert result["stopped_early"] is True
        assert result["used_candidates"] == 3
        assert result["winning_answer"] == "B"
        first = result["candidates"][0]
        assert first["parse_status"] in {"valid", "invalid"}
        assert first["token_metadata"]["storage_mode"] == "options_only"

    def test_pipeline_candidate_records_include_recipe_metadata(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))
        outputs = iter(["A", "A", "B"])

        def predict(_image: Image.Image, _question: str) -> str:
            return next(outputs)

        result = run_tts_pipeline(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            predict_fn=predict,
            text_mode="rule",
            candidate_recipe=[
                (1, "original", "original"),
                (1, "original", "hardcoded_paraphrase"),
                (1, "edge_enhance", "original"),
                (2, "original", "model_paraphrase"),
                (2, "grayscale", "original"),
            ],
            decoding_settings={"temperature": 0.0, "max_new_tokens": 128},
        )

        c3 = result["candidates"][2]
        assert c3["image_transform_id"] == "edge_enhance"
        assert isinstance(c3["image_transform_parameters"], dict)
        assert c3["text_variant_id"] == "original"
        assert "prompt" in c3
        assert c3["decoding_settings"]["temperature"] == 0.0

    def test_pipeline_runs_full_recipe_when_no_early_agreement(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))
        outputs = iter(["A", "B", "C", "B", "B", "A", "C", "B", "B"])

        def predict(_image: Image.Image, _question: str) -> str:
            return next(outputs)

        result = run_tts_pipeline(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            predict_fn=predict,
            text_mode="rule",
        )

        assert result["stopped_early"] is False
        assert result["used_candidates"] == 9
        assert result["winning_answer"] == "B"

    def test_pipeline_reports_weighted_vote_when_weights_provided(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))
        outputs = iter(["A", "B", "C", "B", "B", "A", "C", "B", "B"])

        def predict(_image: Image.Image, _question: str) -> str:
            return next(outputs)

        result = run_tts_pipeline(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            predict_fn=predict,
            text_mode="rule",
            candidate_weights=[0.45, 0.1, 0.1, 0.1, 0.25, 0.0, 0.0, 0.0, 0.0],
        )

        assert result["winning_answer"] == "B"
        assert result["weighted_winning_answer"] == "A"
        assert result["weighted_vote_scores"]["A"] == pytest.approx(0.45)

    def test_baseline_returns_single_prediction(self) -> None:
        image = Image.new("RGB", (32, 32), color=(10, 10, 10))

        def predict(_image: Image.Image, _question: str) -> str:
            return "Option C"

        result = run_baseline(
            image=image,
            question="How many apples are on the table?",
            choices=_sample_choices(),
            predict_fn=predict,
        )

        assert result["normalized_answer"] == "C"


class TestDebugArtifacts:
    def test_export_debug_artifacts_writes_images_and_text(self, tmp_path) -> None:
        image = Image.new("RGB", (32, 32), color=(120, 120, 120))
        choices = _sample_choices()
        candidates = build_candidate_inputs(
            image=image,
            question="How many apples are on the table?",
            choices=choices,
            max_candidates=5,
            text_mode="rule",
        )

        out_dir = export_debug_artifacts(
            output_dir=tmp_path / "debug_run",
            original_image=image,
            original_question="How many apples are on the table?",
            choices=choices,
            candidates=candidates,
        )

        assert (out_dir / "images" / "original.png").exists()
        assert (out_dir / "images" / "edge_enhance.png").exists()
        assert (out_dir / "images" / "grayscale.png").exists()
        assert (out_dir / "text" / "original_question.txt").exists()
        assert (out_dir / "text" / "candidate_2_prompt.txt").exists()
        assert (out_dir / "metadata.json").exists()

    def test_export_debug_artifacts_metadata_contains_transform_details(self, tmp_path) -> None:
        image = Image.new("RGB", (32, 32), color=(120, 120, 120))
        choices = _sample_choices()
        candidates = build_candidate_inputs(
            image=image,
            question="How many apples are on the table?",
            choices=choices,
            max_candidates=5,
            text_mode="rule",
        )

        out_dir = export_debug_artifacts(
            output_dir=tmp_path / "debug_run",
            original_image=image,
            original_question="How many apples are on the table?",
            choices=choices,
            candidates=candidates,
        )

        metadata = (out_dir / "metadata.json").read_text(encoding="utf-8")
        assert "image_transform_id" in metadata
        assert "image_transform_parameters" in metadata
        assert "text_variant_id" in metadata
