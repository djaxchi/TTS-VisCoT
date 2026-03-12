"""I/O helpers for images and JSONL files."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List

import requests
from PIL import Image


def load_image_from_source(source: str) -> Image.Image:
    """Load a PIL image from a local path or HTTP(S) URL.

    Args:
        source: File path or URL string.

    Returns:
        RGB :class:`PIL.Image.Image`.
    """
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(source).convert("RGB")


def save_jsonl(records: List[Dict[str, Any]], path: str | Path) -> Path:
    """Write a list of dicts to a JSONL file.

    Args:
        records: Records to serialise.
        path: Destination file path.

    Returns:
        The resolved :class:`~pathlib.Path` of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return out


def load_jsonl(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    """Yield dicts from a JSONL file line by line.

    Args:
        path: Path to the JSONL file.

    Yields:
        One parsed dict per line.
    """
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
