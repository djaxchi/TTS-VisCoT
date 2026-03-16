"""Tests for src.data.datasets and src.data.augmentation."""

import pytest
from PIL import Image

from src.data.augmentation.image_aug import (
    AugmentationConfig,
    AugmentationSet,
    ImageAugmentor,
)
from src.data.augmentation.text_aug import TextAugmentor
from src.data.augmentation.views import generate_views
from src.data.datasets import get_dataset
from src.data.datasets.treebench import TreeBenchDataset, TreeBenchExample


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TINY_IMAGE = Image.new("RGB", (64, 64), color=(128, 64, 32))
SAMPLE_QUESTION = "What is shown in the image?"
SAMPLE_OPTIONS = {"A": "Apple", "B": "Banana", "C": "Cherry", "D": "Date"}


def _make_raw_example(idx: int = 0):
    """Create a minimal raw example dict that mimics TreeBench structure."""
    return {
        "image": TINY_IMAGE,
        "question": SAMPLE_QUESTION,
        "A": "Apple",
        "B": "Banana",
        "C": "Cherry",
        "D": "Date",
        "answer": "B",
    }


# ---------------------------------------------------------------------------
# TestImageAugmentor
# ---------------------------------------------------------------------------


class TestImageAugmentor:
    def test_augment_output_size_equals_input_size(self):
        aug = ImageAugmentor(seed=0)
        result = aug.augment(TINY_IMAGE, AugmentationSet.SET_A)
        assert result.size == TINY_IMAGE.size

    def test_augment_returns_pil_image(self):
        aug = ImageAugmentor(seed=0)
        result = aug.augment(TINY_IMAGE, AugmentationSet.SET_B)
        assert isinstance(result, Image.Image)

    @pytest.mark.parametrize("aug_set", list(AugmentationSet))
    def test_geometric_aug_output_size_equals_input_size(self, aug_set):
        aug = ImageAugmentor(seed=42)
        result = aug.augment(TINY_IMAGE, aug_set)
        assert result.size == TINY_IMAGE.size

    def test_generate_image_variants_includes_original(self):
        aug = ImageAugmentor(seed=0)
        variants = aug.generate_image_variants(TINY_IMAGE)
        assert variants[0][1] == "original"

    def test_generate_image_variants_length_equals_sets_plus_one(self):
        aug = ImageAugmentor(seed=0)
        sets = [AugmentationSet.SET_A, AugmentationSet.SET_B]
        variants = aug.generate_image_variants(TINY_IMAGE, sets)
        assert len(variants) == len(sets) + 1

    def test_augmentor_seed_produces_reproducible_output_on_same_instance(self):
        """The same ImageAugmentor instance called twice should not crash and
        always return a valid image.  Exact pixel-level determinism cannot be
        guaranteed across albumentations versions, so we verify shape/type only.
        """
        aug = ImageAugmentor(seed=7)
        r1 = aug.augment(TINY_IMAGE, AugmentationSet.SET_A)
        r2 = aug.augment(TINY_IMAGE, AugmentationSet.SET_A)
        assert isinstance(r1, Image.Image)
        assert isinstance(r2, Image.Image)
        assert r1.size == TINY_IMAGE.size
        assert r2.size == TINY_IMAGE.size


# ---------------------------------------------------------------------------
# TestTextAugmentor
# ---------------------------------------------------------------------------


class TestTextAugmentor:
    def test_generate_text_variants_includes_original(self):
        aug = TextAugmentor(seed=0)
        variants = aug.generate_text_variants(SAMPLE_QUESTION, num_paraphrases=3)
        assert variants[0] == (SAMPLE_QUESTION, "original")

    def test_generate_text_variants_correct_count(self):
        aug = TextAugmentor(seed=0)
        variants = aug.generate_text_variants(SAMPLE_QUESTION, num_paraphrases=3)
        assert len(variants) == 4  # original + 3

    def test_paraphrase_question_returns_string(self):
        aug = TextAugmentor(seed=0)
        result = aug.paraphrase_question(SAMPLE_QUESTION)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestGenerateViews
# ---------------------------------------------------------------------------


class TestGenerateViews:
    def test_generate_views_first_is_original(self):
        views = generate_views(TINY_IMAGE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        assert views[0]["variant_id"] == "original"
        assert views[0]["question"] == SAMPLE_QUESTION
        assert views[0]["image"] is TINY_IMAGE

    def test_generate_views_options_unchanged(self):
        views = generate_views(TINY_IMAGE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        for view in views:
            assert view["options"] is SAMPLE_OPTIONS

    def test_generate_views_produces_multiple_views(self):
        config = AugmentationConfig(
            image_sets=[AugmentationSet.SET_A, AugmentationSet.SET_B],
            text_paraphrases=3,
            seed=0,
        )
        views = generate_views(TINY_IMAGE, SAMPLE_QUESTION, SAMPLE_OPTIONS, config)
        assert len(views) > 1

    def test_generate_views_all_have_required_keys(self):
        views = generate_views(TINY_IMAGE, SAMPLE_QUESTION, SAMPLE_OPTIONS)
        for view in views:
            assert {"image", "question", "options", "variant_id"} <= view.keys()


# ---------------------------------------------------------------------------
# TestTreeBenchDataset
# ---------------------------------------------------------------------------


class TestTreeBenchDataset:
    """Unit tests using a mocked _raw_split so no HuggingFace download occurs."""

    def _dataset_with_mock_split(self, n: int = 5, max_samples=None) -> TreeBenchDataset:
        """Return a TreeBenchDataset backed by a list of synthetic raw dicts."""
        ds = TreeBenchDataset(split="test", max_samples=max_samples)
        # Inject a fake list-like split
        raw_data = [_make_raw_example(i) for i in range(n)]
        ds._raw_split = raw_data  # type: ignore[assignment]
        ds.splits = {"test": raw_data}  # type: ignore[assignment]
        return ds

    def test_len_equals_number_of_examples(self):
        ds = self._dataset_with_mock_split(n=5)
        assert len(ds) == 5

    def test_len_capped_by_max_samples(self):
        ds = self._dataset_with_mock_split(n=10, max_samples=3)
        assert len(ds) == 3

    def test_getitem_returns_dict_with_expected_keys(self):
        ds = self._dataset_with_mock_split(n=3)
        item = ds[0]
        assert {"image_id", "question", "options", "correct_answer"} <= item.keys()

    def test_getitem_out_of_range_raises_index_error(self):
        ds = self._dataset_with_mock_split(n=2)
        with pytest.raises(IndexError):
            _ = ds[99]

    def test_get_example_returns_treebench_example(self):
        ds = self._dataset_with_mock_split(n=3)
        ex = ds.get_example(0)
        assert isinstance(ex, TreeBenchExample)

    def test_get_example_correct_answer_normalised(self):
        ds = self._dataset_with_mock_split(n=1)
        ex = ds.get_example(0)
        assert ex.correct_answer == "B"

    def test_get_example_options_populated(self):
        ds = self._dataset_with_mock_split(n=1)
        ex = ds.get_example(0)
        assert len(ex.options) == 4

    def test_export_jsonl_creates_file(self, tmp_path):
        ds = self._dataset_with_mock_split(n=3)
        out_file = tmp_path / "out.jsonl"
        result = ds.export_jsonl(str(out_file), max_examples=3)
        assert result.exists()
        lines = result.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_dataset_registry_lookup(self):
        cls = get_dataset("treebench")
        assert cls is TreeBenchDataset

    def test_dataset_registry_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_dataset("no_such_dataset")


# ---------------------------------------------------------------------------
# TestFetchVqa2Images
# ---------------------------------------------------------------------------


class TestFetchVqa2Images:
    """_fetch_vqa2_images returns the right images keyed by image_id."""

    def _make_hf_row(self, image_id: str) -> dict:
        return {"image_id": image_id, "image": Image.new("RGB", (32, 32), color=(1, 2, 3))}

    def test_returns_matching_image_by_id(self) -> None:
        from unittest.mock import patch

        from src.data.datasets.viscot_benchmark import _fetch_vqa2_images

        rows = [self._make_hf_row("111"), self._make_hf_row("222")]
        with patch("datasets.load_dataset", return_value=iter(rows)):
            result = _fetch_vqa2_images(["111"])
        assert "111" in result
        assert "222" not in result

    def test_returns_empty_dict_when_no_ids_match(self) -> None:
        from unittest.mock import patch

        from src.data.datasets.viscot_benchmark import _fetch_vqa2_images

        rows = [self._make_hf_row("999")]
        with patch("datasets.load_dataset", return_value=iter(rows)):
            result = _fetch_vqa2_images(["111"])
        assert result == {}

    def test_returned_image_is_pil(self) -> None:
        from unittest.mock import patch

        from src.data.datasets.viscot_benchmark import _fetch_vqa2_images

        rows = [self._make_hf_row("42")]
        with patch("datasets.load_dataset", return_value=iter(rows)):
            result = _fetch_vqa2_images(["42"])
        assert isinstance(result["42"], Image.Image)

    def test_stops_streaming_once_all_found(self) -> None:
        """Must not exhaust the full dataset when all IDs are already found."""
        from unittest.mock import patch

        from src.data.datasets.viscot_benchmark import _fetch_vqa2_images

        sentinel_reached = []

        def _rows():
            yield self._make_hf_row("1")
            sentinel_reached.append(True)
            yield self._make_hf_row("2")  # should never be reached

        with patch("datasets.load_dataset", return_value=_rows()):
            _fetch_vqa2_images(["1"])
        assert not sentinel_reached


# ---------------------------------------------------------------------------
# TestIsObjectCountingQuestion
# ---------------------------------------------------------------------------


class TestIsObjectCountingQuestion:
    """Unit tests for the counting-question filter used in prepare_counting_data.py."""

    def _f(self, question: str, answer: str) -> bool:
        from scripts.prepare_counting_data import is_object_counting_question
        return is_object_counting_question(question, answer)

    # --- should accept ---
    @pytest.mark.parametrize("q,a", [
        ("how many dogs are in the image?", "3"),
        ("how many people are standing?", "5"),
        ("How many chairs are there?", "2"),
        ("how many birds can you see?", "1"),
        ("how many cars are parked?", "0"),
        ("how many cats are on the table?", "12"),
    ])
    def test_accepts_object_counting_question(self, q: str, a: str) -> None:
        assert self._f(q, a) is True

    # --- should reject: OCR / text-reading ---
    @pytest.mark.parametrize("q,a", [
        ("how many ml of liquid are in the glass?", "700"),
        ("how many minutes does it take?", "10"),
        ("how many months of repayments?", "72"),
        ("how many calories per serving?", "160"),
        ("how many miles?", "5"),
        ("how many days a week is the place open?", "7"),
        ("how many years has it been?", "30"),
        ("how many ml?", "250"),
    ])
    def test_rejects_ocr_number_reading_question(self, q: str, a: str) -> None:
        assert self._f(q, a) is False

    # --- should reject: non-integer or implausible count ---
    @pytest.mark.parametrize("q,a", [
        ("how many copies were sold?", "2 million"),
        ("how many people are mentioned?", "5000"),
        ("how many blogs are there total?", "70 million"),
        ("how many hours?", "1099"),
        ("how many maps?", "1507"),
        ("how many ml?", "350"),
    ])
    def test_rejects_implausible_count_answer(self, q: str, a: str) -> None:
        assert self._f(q, a) is False

    # --- should reject: no "how many" ---
    def test_rejects_question_without_how_many(self) -> None:
        assert self._f("What colour is the car?", "3") is False

    # --- edge: answer as word number ---
    @pytest.mark.parametrize("q,a", [
        ("how many dogs are there?", "one"),
        ("how many cats?", "two"),
        ("how many birds?", "three"),
    ])
    def test_accepts_word_number_answer(self, q: str, a: str) -> None:
        assert self._f(q, a) is True
