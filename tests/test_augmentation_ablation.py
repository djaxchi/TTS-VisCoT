"""Tests for augmentation ablation analysis.

Verifies flip detection, flip-to-correct classification, and summary
statistics for per-augmentation diversity analysis.
"""

import pytest

from scripts.augmentation_ablation import (
    compute_flip_stats,
    extract_greedy_and_aug_answers,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_row(gt: str, candidates: list[dict]) -> dict:
    """Build a minimal results row."""
    return {
        "gt_answer": gt,
        "answers_all": [gt],
        "candidates": candidates,
    }


def _make_candidate(idx: int, image_aug: str, text_variant: str,
                    temperature: float, answer: str | None) -> dict:
    return {
        "candidate_idx": idx,
        "image_aug": image_aug,
        "text_variant": text_variant,
        "temperature": temperature,
        "answer": answer,
    }


SAMPLE_ROW_FLIP = _make_row("B", [
    _make_candidate(0, "original", "original", 0.0, "A"),          # greedy
    _make_candidate(1, "original", "hardcoded_paraphrase", 0.0, "A"),  # text variant
    _make_candidate(2, "original", "original", 0.7, "C"),          # T=0.7 (skip)
    _make_candidate(3, "edge_enhance", "original", 0.0, "B"),      # flip → correct
    _make_candidate(4, "grayscale", "original", 0.0, "A"),         # no flip
    _make_candidate(5, "jpeg_recompress", "original", 0.0, "C"),   # flip → wrong
    _make_candidate(6, "brightness_contrast", "original", 0.0, "A"),  # no flip
    _make_candidate(7, "rotation_90", "original", 0.0, None),      # null answer
    _make_candidate(8, "edge_enhance", "hardcoded_paraphrase", 0.0, "B"),  # text+aug
])

SAMPLE_ROW_NO_FLIP = _make_row("A", [
    _make_candidate(0, "original", "original", 0.0, "A"),
    _make_candidate(1, "original", "hardcoded_paraphrase", 0.0, "A"),
    _make_candidate(2, "original", "original", 0.7, "A"),
    _make_candidate(3, "edge_enhance", "original", 0.0, "A"),
    _make_candidate(4, "grayscale", "original", 0.0, "A"),
    _make_candidate(5, "jpeg_recompress", "original", 0.0, "A"),
    _make_candidate(6, "brightness_contrast", "original", 0.0, "A"),
    _make_candidate(7, "rotation_90", "original", 0.0, "A"),
    _make_candidate(8, "edge_enhance", "hardcoded_paraphrase", 0.0, "A"),
])


# ── Tests for extract_greedy_and_aug_answers ────────────────────────────────

class TestExtractGreedyAndAugAnswers:

    def test_returns_greedy_answer(self):
        greedy, augs = extract_greedy_and_aug_answers(SAMPLE_ROW_FLIP)
        assert greedy == "A"

    def test_returns_only_image_aug_candidates_at_t0(self):
        """Should exclude: greedy (idx 0), text-only variants, T=0.7 candidates."""
        greedy, augs = extract_greedy_and_aug_answers(SAMPLE_ROW_FLIP)
        aug_names = [a["image_aug"] for a in augs]
        assert "original" not in aug_names  # no original-image candidates
        assert len(augs) == 5  # edge_enhance, grayscale, jpeg, brightness_contrast, rotation_90

    def test_includes_null_answers(self):
        greedy, augs = extract_greedy_and_aug_answers(SAMPLE_ROW_FLIP)
        answers = [a["answer"] for a in augs]
        assert None in answers

    def test_excludes_text_variant_augmentations(self):
        """Candidate 8 has edge_enhance + paraphrase — exclude it to isolate image effect."""
        greedy, augs = extract_greedy_and_aug_answers(SAMPLE_ROW_FLIP)
        for a in augs:
            assert a["text_variant"] == "original"

    def test_greedy_none_when_candidate_0_has_null_answer(self):
        row = _make_row("X", [
            _make_candidate(0, "original", "original", 0.0, None),
            _make_candidate(3, "edge_enhance", "original", 0.0, "Y"),
        ])
        greedy, augs = extract_greedy_and_aug_answers(row)
        assert greedy is None


# ── Tests for compute_flip_stats ────────────────────────────────────────────

class TestComputeFlipStats:

    def test_counts_flips_correctly(self):
        stats = compute_flip_stats([SAMPLE_ROW_FLIP])
        # edge_enhance flips A→B, grayscale no flip, jpeg flips A→C,
        # brightness no flip, rotation null (skip)
        assert stats["edge_enhance"]["flips"] == 1
        assert stats["edge_enhance"]["total"] == 1
        assert stats["jpeg_recompress"]["flips"] == 1
        assert stats["grayscale"]["flips"] == 0
        assert stats["brightness_contrast"]["flips"] == 0

    def test_counts_flip_to_correct(self):
        stats = compute_flip_stats([SAMPLE_ROW_FLIP])
        # edge_enhance: A→B, gt=B → flip to correct
        assert stats["edge_enhance"]["flip_to_correct"] == 1
        # jpeg: A→C, gt=B → flip to wrong
        assert stats["jpeg_recompress"]["flip_to_correct"] == 0

    def test_no_flips_row(self):
        stats = compute_flip_stats([SAMPLE_ROW_NO_FLIP])
        for aug in stats:
            assert stats[aug]["flips"] == 0
            assert stats[aug]["flip_to_correct"] == 0
            assert stats[aug]["total"] >= 1

    def test_skips_null_greedy(self):
        row = _make_row("X", [
            _make_candidate(0, "original", "original", 0.0, None),
            _make_candidate(3, "edge_enhance", "original", 0.0, "Y"),
        ])
        stats = compute_flip_stats([row])
        # Should skip this row entirely since greedy is null
        assert stats["edge_enhance"]["total"] == 0

    def test_skips_null_aug_answer(self):
        stats = compute_flip_stats([SAMPLE_ROW_FLIP])
        # rotation_90 has null answer — should not count as total
        assert stats["rotation_90"]["total"] == 0

    def test_multiple_rows_accumulate(self):
        stats = compute_flip_stats([SAMPLE_ROW_FLIP, SAMPLE_ROW_NO_FLIP])
        # edge_enhance: 1 flip from row 1, 0 from row 2 = 1 flip, 2 total
        assert stats["edge_enhance"]["total"] == 2
        assert stats["edge_enhance"]["flips"] == 1

    def test_flip_from_correct_to_wrong_tracked(self):
        stats = compute_flip_stats([SAMPLE_ROW_FLIP])
        # jpeg: A→C, gt=B, greedy was wrong, flip is also wrong
        assert stats["jpeg_recompress"]["flip_from_correct"] == 0

    def test_flip_from_correct_to_wrong_when_greedy_correct(self):
        row = _make_row("A", [
            _make_candidate(0, "original", "original", 0.0, "A"),  # greedy correct
            _make_candidate(3, "edge_enhance", "original", 0.0, "B"),  # flip away
        ])
        stats = compute_flip_stats([row])
        assert stats["edge_enhance"]["flip_from_correct"] == 1
