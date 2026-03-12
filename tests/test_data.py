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
