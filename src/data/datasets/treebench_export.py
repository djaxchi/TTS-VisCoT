"""Utility to export a small set of TreeBench samples to local files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.data.datasets.treebench import TreeBenchDataset


def load_exported_sample(
    export_dir: str | Path,
    index: int = 0,
) -> Optional[Tuple[Image.Image, str, Dict[str, str], str]]:
    """Load one sample from a local TreeBench export folder.

    Args:
        export_dir: Folder containing ``metadata.jsonl`` and ``images/``.
        index: 0-based line index in metadata.jsonl.

    Returns:
        Tuple ``(image, question, options, image_id)`` or ``None`` if missing/invalid.
    """
    export_path = Path(export_dir)
    meta_path = export_path / "metadata.jsonl"
    if not meta_path.exists():
        return None

    try:
        lines = meta_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    if index < 0 or index >= len(lines):
        return None

    try:
        row = json.loads(lines[index])
        image_rel = row["image_path"]
        image_path = export_path / image_rel
        image = Image.open(image_path).convert("RGB")
        question = str(row["question"])
        options = dict(row["options"])
        image_id = str(row.get("image_id", f"exported_{index}"))
        return image, question, options, image_id
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def export_treebench_samples(
    output_dir: str | Path,
    n: int = 5,
    split: str = "train",
    dataset_cls: type = TreeBenchDataset,
) -> List[Path]:
    """Export first *n* TreeBench samples as PNG + JSONL metadata.

    Args:
        output_dir: Destination folder.
        n: Number of samples to export.
        split: Dataset split name.
        dataset_cls: Injectable dataset class for testing.

    Returns:
        List of saved image paths.
    """
    out = Path(output_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = dataset_cls(split=split, max_samples=n)
    try:
        ds.load()
    except ValueError as exc:
        if "Unknown split" in str(exc) and split != "train":
            ds = dataset_cls(split="train", max_samples=n)
            ds.load()
        else:
            raise

    n_to_export = min(n, len(ds))
    saved_paths: List[Path] = []

    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for i in range(n_to_export):
            ex = ds.get_example(i)
            if ex is None:
                continue

            image_path = images_dir / f"{ex.image_id}.png"
            ex.image.convert("RGB").save(image_path)
            saved_paths.append(image_path)

            row: dict[str, Any] = {
                "index": i,
                "image_id": ex.image_id,
                "question": ex.question,
                "options": ex.options,
                "correct_answer": ex.correct_answer,
                "image_path": str(image_path.relative_to(out)).replace("\\", "/"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return saved_paths
