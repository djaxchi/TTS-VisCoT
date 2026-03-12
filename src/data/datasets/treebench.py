"""TreeBench dataset loader.

Wraps the ``HaochenWang/TreeBench`` Hugging Face dataset and exposes it
through the :class:`BaseDataset` interface.
"""

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from PIL import Image
from pydantic import BaseModel, Field

from .base import BaseDataset


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box with an optional class label."""

    x: float
    y: float
    width: float
    height: float
    label: Optional[str] = None


class TreeBenchExample(BaseModel):
    """A single TreeBench example."""

    image_id: str
    image: Any  # PIL Image at runtime; typed as Any for Pydantic compatibility
    question: str
    options: Dict[str, str]
    correct_answer: str
    bboxes: List[BoundingBox] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class TreeBenchDataset(BaseDataset):
    """Loader for the TreeBench visual QA benchmark.

    Example::

        dataset = TreeBenchDataset(split="test", max_samples=100)
        dataset.load()
        example = dataset[0]

    Args:
        split: HuggingFace split name (``None`` loads all splits).
        max_samples: Cap on the number of examples returned.
        cache_dir: Optional local cache directory for HuggingFace datasets.
    """

    HF_REPO = "HaochenWang/TreeBench"

    def __init__(
        self,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(split=split, max_samples=max_samples, cache_dir=cache_dir)
        self._raw_split: Optional[Dataset] = None
        self.splits: Dict[str, Dataset] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Dataset]:
        """Download / load the dataset from HuggingFace Hub.

        Returns:
            Mapping of split name → ``datasets.Dataset``.
        """
        try:
            if self.split:
                raw = load_dataset(self.HF_REPO, split=self.split, cache_dir=self.cache_dir)
                self.splits = {self.split: raw}
            else:
                raw = load_dataset(self.HF_REPO, cache_dir=self.cache_dir)
                self.splits = dict(raw)
        except Exception as exc:
            # Fallback: load only the test split
            raw = load_dataset(self.HF_REPO, split="test", cache_dir=self.cache_dir)
            self.splits = {"test": raw}

        # Store the active split for __len__ / __getitem__
        target_split = self.split or "test"
        self._raw_split = self.splits.get(target_split, next(iter(self.splits.values())))
        return self.splits

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._raw_split is None:
            return 0
        n = len(self._raw_split)
        return min(n, self.max_samples) if self.max_samples is not None else n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._raw_split is None:
            raise RuntimeError("Call .load() before indexing the dataset.")
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}.")
        example = self._parse_example(self._raw_split[idx], idx)
        return {
            "image_id": example.image_id,
            "image": example.image,
            "question": example.question,
            "options": example.options,
            "correct_answer": example.correct_answer,
            "bboxes": [b.model_dump() for b in example.bboxes],
            "metadata": example.metadata,
        }

    # Convenience alias used by the smoke-test / experiments
    def get_example(self, idx: int, split: Optional[str] = None) -> Optional[TreeBenchExample]:
        """Return example ``idx`` as a :class:`TreeBenchExample` object.

        Args:
            idx: Zero-based index.
            split: If given, look up from that split instead of the active one.

        Returns:
            Parsed :class:`TreeBenchExample` or ``None`` on error.
        """
        try:
            src = self.splits.get(split) if split else self._raw_split
            if src is None:
                return None
            if idx >= len(src):
                return None
            return self._parse_example(src[idx], idx)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(
        self,
        output_path: str,
        split: Optional[str] = None,
        max_examples: Optional[int] = None,
    ) -> Path:
        """Export examples to a JSONL file (images excluded).

        Args:
            output_path: Destination file path.
            split: Specific split to export; defaults to the active split.
            max_examples: Maximum number of examples to write.

        Returns:
            :class:`~pathlib.Path` of the written file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        src = self.splits.get(split) if split else self._raw_split
        if src is None:
            raise RuntimeError("Dataset not loaded.")
        n = len(src)
        if max_examples:
            n = min(n, max_examples)
        with out.open("w") as fh:
            for i in range(n):
                ex = self._parse_example(src[i], i)
                if ex:
                    fh.write(
                        json.dumps(
                            {
                                "image_id": ex.image_id,
                                "question": ex.question,
                                "options": ex.options,
                                "correct_answer": ex.correct_answer,
                                "gt_bboxes": [b.model_dump() for b in ex.bboxes],
                                "metadata": ex.metadata,
                            }
                        )
                        + "\n"
                    )
        return out

    def get_statistics(self, split: Optional[str] = None) -> Dict[str, Any]:
        """Compute summary statistics for a split (sampled at most 1 000 examples).

        Args:
            split: Split to analyse; defaults to the active split.

        Returns:
            Dict with keys ``total_examples``, ``answer_distribution``,
            ``has_bboxes``, and ``avg_options``.
        """
        src = self.splits.get(split) if split else self._raw_split
        if src is None:
            return {}
        stats: Dict[str, Any] = {
            "total_examples": len(src),
            "answer_distribution": {},
            "has_bboxes": 0,
            "avg_options": 0.0,
        }
        option_counts: List[int] = []
        for i in range(min(len(src), 1000)):
            ex = self._parse_example(src[i], i)
            if ex:
                ans = ex.correct_answer
                stats["answer_distribution"][ans] = (
                    stats["answer_distribution"].get(ans, 0) + 1
                )
                if ex.bboxes:
                    stats["has_bboxes"] += 1
                option_counts.append(len(ex.options))
        if option_counts:
            stats["avg_options"] = float(np.mean(option_counts))
        return stats

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    def _parse_example(self, raw: Dict, idx: int) -> TreeBenchExample:
        image = raw.get("image") or raw.get("img")
        if isinstance(image, str):
            image = self._decode_base64_image(image)

        question = raw.get("question") or raw.get("query") or ""
        options = self._extract_options(raw)
        correct_answer = self._extract_answer(raw)
        bboxes = self._extract_bboxes(raw)
        metadata = {
            k: raw.get(k)
            for k in ("protocol", "task_type", "category", "difficulty")
            if raw.get(k) is not None
        }
        return TreeBenchExample(
            image_id=f"example_{idx}",
            image=image,
            question=question,
            options=options,
            correct_answer=correct_answer,
            bboxes=bboxes,
            metadata=metadata,
        )

    def _decode_base64_image(self, image_str: str) -> Optional[Image.Image]:
        try:
            if image_str.startswith("data:"):
                image_str = image_str.split(",", 1)[1]
            data = base64.b64decode(image_str)
            img = Image.open(io.BytesIO(data))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            return img
        except Exception:
            return None

    def _extract_options(self, raw: Dict) -> Dict[str, str]:
        if "options" in raw:
            opts = raw["options"]
            if isinstance(opts, dict):
                return opts
            if isinstance(opts, list):
                return {chr(65 + i): str(o) for i, o in enumerate(opts)}
        for letter in "ABCDEFGHIJK":
            if raw.get(letter) is not None:
                # Build dict lazily from per-letter fields
                return {
                    k: str(raw[k]).strip()
                    for k in "ABCDEFGHIJK"
                    if raw.get(k) is not None
                }
        if "choices" in raw:
            choices = raw["choices"]
            if isinstance(choices, list):
                return {chr(65 + i): str(c) for i, c in enumerate(choices)}
            if isinstance(choices, dict):
                return choices
        return {}

    def _extract_answer(self, raw: Dict) -> str:
        for field in ("answer", "correct_answer", "label", "ground_truth"):
            if field in raw:
                return self._normalise_raw_answer(raw[field])
        return ""

    @staticmethod
    def _normalise_raw_answer(answer: Any) -> str:
        if isinstance(answer, int):
            return chr(65 + answer)
        s = str(answer).strip().upper()
        if len(s) == 1 and s in "ABCDEFGH":
            return s
        for letter in "ABCDEFGH":
            if letter in s:
                return letter
        return s

    def _extract_bboxes(self, raw: Dict) -> List[BoundingBox]:
        for field in ("bboxes", "bounding_boxes", "boxes", "bbox"):
            data = raw.get(field)
            if data:
                result = []
                for bbox in data:
                    if isinstance(bbox, dict):
                        result.append(BoundingBox(**bbox))
                    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        result.append(
                            BoundingBox(
                                x=bbox[0],
                                y=bbox[1],
                                width=bbox[2],
                                height=bbox[3],
                                label=bbox[4] if len(bbox) > 4 else None,
                            )
                        )
                return result
        return []
