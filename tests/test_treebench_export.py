"""Tests for TreeBench sample export utility."""

from __future__ import annotations

from pathlib import Path
import json

from PIL import Image

from src.data.datasets.treebench_export import export_treebench_samples, load_exported_sample


class _FakeDataset:
    def __init__(self, split: str = "test", max_samples: int | None = None) -> None:
        self.split = split
        self.max_samples = max_samples
        self._loaded = False

    def load(self):
        self._loaded = True
        return {"test": []}

    def __len__(self) -> int:
        return 3

    def get_example(self, idx: int):
        if not self._loaded:
            raise RuntimeError("not loaded")
        return type(
            "Example",
            (),
            {
                "image_id": f"img_{idx}",
                "image": Image.new("RGB", (32, 24), color=(20 + idx, 30, 40)),
                "question": f"Question {idx}",
                "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
                "correct_answer": "A",
            },
        )


def test_export_treebench_samples_writes_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "samples"
    saved = export_treebench_samples(
        output_dir=out_dir,
        n=2,
        dataset_cls=_FakeDataset,
    )

    assert len(saved) == 2
    assert (out_dir / "images" / "img_0.png").exists()
    assert (out_dir / "images" / "img_1.png").exists()
    assert (out_dir / "metadata.jsonl").exists()


def test_export_treebench_samples_caps_to_dataset_len(tmp_path: Path) -> None:
    out_dir = tmp_path / "samples"
    saved = export_treebench_samples(
        output_dir=out_dir,
        n=10,
        dataset_cls=_FakeDataset,
    )

    assert len(saved) == 3


def test_load_exported_sample_reads_image_and_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "samples"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True)

    image_path = images_dir / "example_0.png"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(image_path)

    row = {
        "index": 0,
        "image_id": "example_0",
        "question": "Q?",
        "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
        "correct_answer": "A",
        "image_path": "images/example_0.png",
    }
    (out_dir / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_exported_sample(out_dir, index=0)
    assert loaded is not None
    image, question, options, image_id = loaded
    assert image.size == (16, 16)
    assert question == "Q?"
    assert options["A"] == "x"
    assert image_id == "example_0"


def test_load_exported_sample_returns_none_when_missing(tmp_path: Path) -> None:
    loaded = load_exported_sample(tmp_path / "does_not_exist", index=0)
    assert loaded is None
