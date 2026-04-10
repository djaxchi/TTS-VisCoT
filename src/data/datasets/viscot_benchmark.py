"""Hard-bench dataset loader.

Reads curated question/answer JSONL files from ``data/hard_bench/`` and
fetches images on-demand from public HuggingFace mirrors:

- ``mmmu_pro``  → ``MMMU/MMMU_Pro`` (standard 10 options, test split)
- ``ocrbench``  → ``echo840/OCRBench`` (test split)
- ``chartqa``   → ``lmms-lab/ChartQA`` (test split)
- ``gqa``       → ``lmms-lab/GQA`` (val_balanced_images, retained for compatibility)

The JSONL rows have fields::

    question_id, question, answer, image_id, image_source

Images are downloaded once, saved to ``data/hard_bench/images/<source>/``
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
_BENCHMARK_DIR = _REPO_ROOT / "data" / "hard_bench"
_IMAGE_CACHE_DIR = _REPO_ROOT / "data" / "hard_bench" / "images"


# ---------------------------------------------------------------------------
# Disk image cache helpers
# ---------------------------------------------------------------------------


def _disk_path(source: str, image_id: str) -> Path:
    return _IMAGE_CACHE_DIR / source / f"{image_id}.jpg"


def _save_to_disk(source: str, image_id: str, pil: Image.Image) -> None:
    path = _disk_path(source, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, format="JPEG", quality=95)


def _load_from_disk(
    source: str, image_ids: List[str]
) -> tuple[Dict[str, Image.Image], List[str]]:
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
# Public API
# ---------------------------------------------------------------------------


def load_task(
    task: str,
    n: Optional[int] = None,
    offset: int = 0,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Load *n* samples for *task* with images attached.

    Images are served from a local disk cache (``data/hard_bench/images/``).
    If an image is not cached it is fetched from HuggingFace and saved for
    future use.

    Args:
        task: One of ``"vqa"``, ``"counting"``, ``"ocr"``.
        n: Number of samples to return. ``None`` returns all 100.
        offset: Skip the first *offset* rows.
        seed: Unused (order is deterministic from the JSONL file).

    Returns:
        List of dicts with keys ``question_id``, ``question``, ``answer``,
        ``image_id``, ``image_source``, ``image`` (PIL.Image).

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        ValueError: If *task* is not recognised.
    """
    task = task.lower()
    fname_map = {
        "vqa": "vqa_100.jsonl",
        "counting": "counting_100.jsonl",
        "ocr": "ocr_100.jsonl",
    }
    if task not in fname_map:
        raise ValueError(f"Unknown task {task!r}. Expected one of {list(fname_map)}")

    jsonl_path = _BENCHMARK_DIR / fname_map[task]
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {jsonl_path}\n"
            "Run scripts/prepare_hard_bench.py to generate it."
        )

    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if offset:
        rows = rows[offset:]
    if n is not None:
        rows = rows[:n]

    # Group by image_source for batched fetching.
    by_source: Dict[str, List[str]] = {}
    for row in rows:
        src = row["image_source"]
        by_source.setdefault(src, []).append(row["image_id"])

    image_cache: Dict[str, Image.Image] = {}
    for source, ids in by_source.items():
        image_cache.update(_fetch_images(source, ids))

    result: List[Dict[str, Any]] = []
    missing_count = 0
    for row in rows:
        img = image_cache.get(row["image_id"])
        if img is None:
            missing_count += 1
            continue
        result.append({**row, "image": img})

    if missing_count:
        logger.warning(
            "{}/{} samples dropped (image not found).", missing_count, len(rows)
        )

    return result


# ---------------------------------------------------------------------------
# Image fetchers — disk cache first, HF stream fallback
# ---------------------------------------------------------------------------


def _fetch_images(source: str, image_ids: List[str]) -> Dict[str, Image.Image]:
    if source == "mmmu_pro":
        return _fetch_mmmu_pro_images(image_ids)
    if source == "ocrbench":
        return _fetch_ocrbench_images(image_ids)
    if source == "chartqa":
        return _fetch_chartqa_images(image_ids)
    if source == "gqa":
        return _fetch_gqa_images(image_ids)
    raise ValueError(f"Unknown image_source {source!r}")


def _fetch_mmmu_pro_images(image_ids: List[str]) -> Dict[str, Image.Image]:
    """Fetch MMMU-Pro images by question ID (embedded PIL in dataset)."""
    found, needed_list = _load_from_disk("mmmu_pro", image_ids)
    if not needed_list:
        logger.info("MMMU-Pro: all {} images loaded from disk cache.", len(found))
        return found

    logger.info(
        "MMMU-Pro: {}/{} images on disk; fetching {} from MMMU/MMMU_Pro…",
        len(found), len(image_ids), len(needed_list),
    )
    from datasets import load_dataset  # type: ignore

    needed = set(needed_list)
    ds = load_dataset(
        "MMMU/MMMU_Pro", "standard (10 options)", split="test", streaming=True
    )
    for row in ds:
        if not needed:
            break
        qid = str(row.get("id", ""))
        if qid not in needed:
            continue
        pil = _to_pil(row.get("image_1"))
        if pil is not None:
            found[qid] = pil
            needed.discard(qid)
            _save_to_disk("mmmu_pro", qid, pil)

    logger.info("MMMU-Pro: retrieved {}/{} images.", len(found), len(image_ids))
    return found


def _fetch_ocrbench_images(image_ids: List[str]) -> Dict[str, Image.Image]:
    """Fetch OCRBench images by sequential dataset index."""
    found, needed_list = _load_from_disk("ocrbench", image_ids)
    if not needed_list:
        logger.info("OCRBench: all {} images loaded from disk cache.", len(found))
        return found

    logger.info(
        "OCRBench: {}/{} images on disk; fetching {} from echo840/OCRBench…",
        len(found), len(image_ids), len(needed_list),
    )
    from datasets import load_dataset  # type: ignore

    needed = set(needed_list)
    needed_ints = {int(i) for i in needed}
    ds = load_dataset("echo840/OCRBench", split="test", streaming=True)
    for idx, row in enumerate(ds):
        if not needed:
            break
        if idx not in needed_ints:
            continue
        iid = str(idx)
        pil = _to_pil(row.get("image"))
        if pil is not None:
            found[iid] = pil
            needed.discard(iid)
            needed_ints.discard(idx)
            _save_to_disk("ocrbench", iid, pil)

    logger.info("OCRBench: retrieved {}/{} images.", len(found), len(image_ids))
    return found


def _fetch_chartqa_images(image_ids: List[str]) -> Dict[str, Image.Image]:
    """Fetch ChartQA images by sequential dataset index."""
    found, needed_list = _load_from_disk("chartqa", image_ids)
    if not needed_list:
        logger.info("ChartQA: all {} images loaded from disk cache.", len(found))
        return found

    logger.info(
        "ChartQA: {}/{} images on disk; fetching {} from lmms-lab/ChartQA…",
        len(found), len(image_ids), len(needed_list),
    )
    from datasets import load_dataset  # type: ignore

    needed = set(needed_list)
    needed_ints = {int(i) for i in needed}
    ds = load_dataset("lmms-lab/ChartQA", split="test", streaming=True)
    for idx, row in enumerate(ds):
        if not needed:
            break
        if idx not in needed_ints:
            continue
        iid = str(idx)
        pil = _to_pil(row.get("image"))
        if pil is not None:
            found[iid] = pil
            needed.discard(iid)
            needed_ints.discard(idx)
            _save_to_disk("chartqa", iid, pil)

    logger.info("ChartQA: retrieved {}/{} images.", len(found), len(image_ids))
    return found


def _fetch_gqa_images(image_ids: List[str]) -> Dict[str, Image.Image]:
    """Fetch GQA images (retained for backward-compatibility)."""
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


def _to_pil(img_data: Any) -> Optional[Image.Image]:
    if img_data is None:
        return None
    if isinstance(img_data, bytes):
        return Image.open(BytesIO(img_data)).convert("RGB")
    if hasattr(img_data, "convert"):
        return img_data.convert("RGB")
    return None
