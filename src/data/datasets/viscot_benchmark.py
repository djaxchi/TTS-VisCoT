"""VisCoT benchmark dataset loader.

Reads curated question/answer JSONL files from ``data/benchmark/`` and
fetches images on-demand from public HuggingFace mirrors:

- ``gqa``     → ``lmms-lab/GQA``  (val_balanced_images config)
- ``textvqa`` → ``lmms-lab/textvqa`` (validation split)

The JSONL rows have fields::

    question_id, question, answer, image_id, image_source

Images are downloaded once, saved to ``data/benchmark/images/<source>/``
and reloaded from disk on subsequent runs (no network required).
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BENCHMARK_DIR = _REPO_ROOT / "data" / "benchmark"
_IMAGE_CACHE_DIR = _BENCHMARK_DIR / "images"

_HF_IMAGE_CONFIGS = {
    "gqa": ("lmms-lab/GQA", "val_balanced_images", "val"),
    "textvqa": ("lmms-lab/textvqa", None, "validation"),
    # counting questions come from textvqa training data
    "textvqa_train": ("lmms-lab/textvqa", None, "train"),
    # VQAv2 validation — used for object-counting questions.
    "vqa2": ("lmms-lab/VQAv2", None, "validation"),
}


# ---------------------------------------------------------------------------
# Disk image cache helpers
# ---------------------------------------------------------------------------


def _disk_path(source: str, image_id: str) -> Path:
    return _IMAGE_CACHE_DIR / source / f"{image_id}.jpg"


def _save_to_disk(source: str, image_id: str, pil: Image.Image) -> None:
    path = _disk_path(source, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, format="JPEG", quality=95)


def _load_from_disk(source: str, image_ids: List[str]) -> tuple[Dict[str, Image.Image], List[str]]:
    """Return (found_dict, still_needed_ids) from the local disk cache."""
    found: Dict[str, Image.Image] = {}
    missing: List[str] = []
    for iid in image_ids:
        p = _disk_path(source, iid)
        if p.exists():
            try:
                found[iid] = Image.open(p).convert("RGB")
            except Exception:
                missing.append(iid)
        else:
            missing.append(iid)
    return found, missing


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_task(
    task: str,
    n: Optional[int] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Load *n* samples for *task* with images attached.

    Images are served from a local disk cache (``data/benchmark/images/``).
    If an image is not cached yet it is fetched from HuggingFace and saved
    for future use.

    Args:
        task: One of ``"vqa"``, ``"counting"``, ``"ocr"``.
        n: Number of samples to return. ``None`` returns all 100.
        seed: Unused (order is deterministic from the JSONL file).

    Returns:
        List of dicts with keys ``question_id``, ``question``, ``answer``,
        ``image_id``, ``image_source``, ``image`` (PIL.Image).

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        ValueError: If *task* is not recognised.
    """
    task = task.lower()
    fname_map = {"vqa": "vqa_100.jsonl", "counting": "counting_100.jsonl", "ocr": "ocr_100.jsonl"}
    if task not in fname_map:
        raise ValueError(f"Unknown task {task!r}. Expected one of {list(fname_map)}")

    jsonl_path = _BENCHMARK_DIR / fname_map[task]
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {jsonl_path}\n"
            "Run scripts/prepare_benchmark_data.py to generate it."
        )

    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    if n is not None:
        rows = rows[:n]

    # Group rows by image_source for batched image fetching.
    by_source: Dict[str, List[str]] = {}
    for row in rows:
        src = row["image_source"]
        by_source.setdefault(src, [])
        by_source[src].append(row["image_id"])

    image_cache: Dict[str, Image.Image] = {}
    for source, ids in by_source.items():
        image_cache.update(_fetch_images(source, ids))

    # Attach images; drop rows whose image wasn't found.
    result: List[Dict[str, Any]] = []
    missing = 0
    for row in rows:
        img = image_cache.get(row["image_id"])
        if img is None:
            missing += 1
            continue
        result.append({**row, "image": img})

    if missing:
        logger.warning("{}/{} samples dropped (image not found).", missing, len(rows))

    return result


# ---------------------------------------------------------------------------
# Internal image fetchers (disk cache → HF stream fallback)
# ---------------------------------------------------------------------------


def _fetch_images(source: str, image_ids: List[str]) -> Dict[str, Image.Image]:
    """Fetch images by ID: try disk cache first, then HuggingFace stream."""
    if source == "gqa":
        return _fetch_gqa_images(image_ids)
    if source in ("textvqa", "textvqa_train"):
        split = "train" if source == "textvqa_train" else "validation"
        return _fetch_textvqa_images(image_ids, split=split)
    if source == "vqa2":
        return _fetch_vqa2_images(image_ids)
    raise ValueError(f"Unknown image_source {source!r}")


def _fetch_gqa_images(image_ids: List[str]) -> Dict[str, Image.Image]:
    found, needed_list = _load_from_disk("gqa", image_ids)
    if not needed_list:
        logger.info("GQA: all {} images loaded from disk cache.", len(found))
        return found

    logger.info(
        "GQA: {}/{} images on disk; fetching {} from lmms-lab/GQA…",
        len(found), len(image_ids), len(needed_list),
    )
    from datasets import load_dataset  # type: ignore

    needed = set(needed_list)
    ds = load_dataset("lmms-lab/GQA", "val_balanced_images", split="val", streaming=True)
    for row in ds:
        if not needed:
            break
        iid = str(row.get("id", row.get("imageId", "")))
        if iid not in needed:
            continue
        pil = _to_pil(row.get("image") or row.get("image_bytes"))
        if pil is not None:
            found[iid] = pil
            needed.discard(iid)
            _save_to_disk("gqa", iid, pil)

    logger.info("GQA: retrieved {}/{} images.", len(found), len(image_ids))
    return found


def _fetch_textvqa_images(image_ids: List[str], split: str = "validation") -> Dict[str, Image.Image]:
    source_key = "textvqa_train" if split == "train" else "textvqa"
    found, needed_list = _load_from_disk(source_key, image_ids)
    if not needed_list:
        logger.info("TextVQA: all {} images loaded from disk cache.", len(found))
        return found

    logger.info(
        "TextVQA: {}/{} images on disk; fetching {} from lmms-lab/textvqa ({})…",
        len(found), len(image_ids), len(needed_list), split,
    )
    from datasets import load_dataset  # type: ignore

    needed = set(needed_list)
    ds = load_dataset("lmms-lab/textvqa", split=split, streaming=True)
    for row in ds:
        if not needed:
            break
        iid = str(row.get("image_id", ""))
        if iid not in needed:
            continue
        pil = _to_pil(row.get("image"))
        if pil is not None:
            found[iid] = pil
            needed.discard(iid)
            _save_to_disk(source_key, iid, pil)

    logger.info("TextVQA: retrieved {}/{} images.", len(found), len(image_ids))
    return found


def _fetch_vqa2_images(image_ids: List[str]) -> Dict[str, Image.Image]:
    found, needed_list = _load_from_disk("vqa2", image_ids)
    if not needed_list:
        logger.info("VQAv2: all {} images loaded from disk cache.", len(found))
        return found

    logger.info(
        "VQAv2: {}/{} images on disk; fetching {} from lmms-lab/VQAv2…",
        len(found), len(image_ids), len(needed_list),
    )
    from datasets import load_dataset  # type: ignore

    needed = set(needed_list)
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    for row in ds:
        if not needed:
            break
        iid = str(row.get("image_id", ""))
        if iid not in needed:
            continue
        pil = _to_pil(row.get("image"))
        if pil is not None:
            found[iid] = pil
            needed.discard(iid)
            _save_to_disk("vqa2", iid, pil)
            if not needed:
                break

    logger.info("VQAv2: retrieved {}/{} images.", len(found), len(image_ids))
    return found


def _to_pil(img_data: Any) -> Optional[Image.Image]:
    if img_data is None:
        return None
    if isinstance(img_data, bytes):
        return Image.open(BytesIO(img_data)).convert("RGB")
    if hasattr(img_data, "convert"):
        return img_data.convert("RGB")
    return None
