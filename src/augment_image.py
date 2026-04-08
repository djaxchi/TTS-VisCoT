"""Image variation helpers for TreeBench test-time scaling."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict

from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class ImageVariationConfig:
    """Configuration for deterministic TreeBench image variants.

    Attributes:
        preset: Strength preset name (``conservative``, ``moderate``, ``strong``).
        enable_brightness_contrast: Enable strong brightness/contrast transform.
        enable_jpeg_recompress: Enable JPEG degradation transform.
        enable_grayscale: Enable grayscale transform.
        enable_edge_enhance: Enable sharpen/edge-enhance transform.
        enable_binary_bw: Enable optional binary black/white transform.
        enable_rotation: Enable optional rotation ablation (disabled by default).
        rotation_degrees: Optional rotation degrees if rotation is enabled.
    """

    preset: str = "strong"
    enable_brightness_contrast: bool = True
    enable_jpeg_recompress: bool = True
    enable_grayscale: bool = True
    enable_edge_enhance: bool = True
    enable_binary_bw: bool = False
    enable_rotation: bool = True
    rotation_degrees: tuple[int, ...] = (90,)


_PRESET_PARAMS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "brightness": 1.18,
        "contrast": 1.20,
        "jpeg_quality": 68,
        "edge_percent": 180,
        "edge_radius": 1.3,
        "bw_threshold": 125,
    },
    "moderate": {
        "brightness": 1.45,
        "contrast": 1.50,
        "jpeg_quality": 45,
        "edge_percent": 260,
        "edge_radius": 1.8,
        "bw_threshold": 120,
    },
    "strong": {
        "brightness": 1.80,
        "contrast": 1.85,
        "jpeg_quality": 28,
        "edge_percent": 340,
        "edge_radius": 2.2,
        "bw_threshold": 118,
    },
}


def _adjust_brightness_contrast(image: Image.Image, brightness: float, contrast: float) -> Image.Image:
    out = ImageEnhance.Brightness(image).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    return out


def _jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L").convert("RGB")


def _edge_enhance(image: Image.Image, radius: float, percent: int) -> Image.Image:
    sharp = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
    return sharp.filter(ImageFilter.EDGE_ENHANCE_MORE)


def _binary_bw(image: Image.Image, threshold: int) -> Image.Image:
    gray = image.convert("L")
    bw = gray.point(lambda p: 255 if p >= threshold else 0, mode="1")
    return bw.convert("RGB")


def _rotate(image: Image.Image, degrees: int) -> Image.Image:
    """Rotate image with full content preserved; canvas expands to fit (no cropping, no scaling)."""
    return image.rotate(degrees, expand=True)


def _preset_params(config: ImageVariationConfig) -> Dict[str, Any]:
    return _PRESET_PARAMS.get(config.preset, _PRESET_PARAMS["strong"])


def generate_image_variant_specs(
    image: Image.Image,
    config: ImageVariationConfig | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate image variants with transform metadata for auditing.

    Returns:
        Mapping from ``transform_id`` to a spec dict containing:
        ``image``, ``transform_id``, ``parameters``, and ``preset``.
    """
    cfg = config or ImageVariationConfig()
    base = image.convert("RGB")
    params = _preset_params(cfg)

    specs: Dict[str, Dict[str, Any]] = {
        "original": {
            "image": base,
            "transform_id": "original",
            "parameters": {},
            "preset": cfg.preset,
        }
    }

    if cfg.enable_brightness_contrast:
        specs["brightness_contrast"] = {
            "image": _adjust_brightness_contrast(
                base,
                brightness=float(params["brightness"]),
                contrast=float(params["contrast"]),
            ),
            "transform_id": "brightness_contrast",
            "parameters": {
                "brightness": float(params["brightness"]),
                "contrast": float(params["contrast"]),
            },
            "preset": cfg.preset,
        }

    if cfg.enable_jpeg_recompress:
        specs["jpeg_recompress"] = {
            "image": _jpeg_recompress(base, quality=int(params["jpeg_quality"])),
            "transform_id": "jpeg_recompress",
            "parameters": {"quality": int(params["jpeg_quality"])},
            "preset": cfg.preset,
        }

    if cfg.enable_grayscale:
        specs["grayscale"] = {
            "image": _to_grayscale(base),
            "transform_id": "grayscale",
            "parameters": {"mode": "L->RGB"},
            "preset": cfg.preset,
        }

    if cfg.enable_edge_enhance:
        specs["edge_enhance"] = {
            "image": _edge_enhance(
                base,
                radius=float(params["edge_radius"]),
                percent=int(params["edge_percent"]),
            ),
            "transform_id": "edge_enhance",
            "parameters": {
                "radius": float(params["edge_radius"]),
                "percent": int(params["edge_percent"]),
                "threshold": 2,
            },
            "preset": cfg.preset,
        }

    if cfg.enable_binary_bw:
        specs["binary_bw"] = {
            "image": _binary_bw(base, threshold=int(params["bw_threshold"])),
            "transform_id": "binary_bw",
            "parameters": {"threshold": int(params["bw_threshold"])},
            "preset": cfg.preset,
        }

    if cfg.enable_rotation:
        for deg in cfg.rotation_degrees:
            key = f"rotation_{deg}"
            specs[key] = {
                "image": _rotate(base, degrees=int(deg)),
                "transform_id": "rotation",
                "parameters": {"degrees": int(deg)},
                "preset": cfg.preset,
            }

    # Backward-compatible aliases used by the existing 3->5 pipeline recipe.
    if "edge_enhance" in specs:
        specs["image_variation_1"] = {
            "image": specs["edge_enhance"]["image"],
            "transform_id": "edge_enhance",
            "parameters": dict(specs["edge_enhance"]["parameters"]),
            "preset": specs["edge_enhance"]["preset"],
        }
    elif "brightness_contrast" in specs:
        specs["image_variation_1"] = {
            "image": specs["brightness_contrast"]["image"],
            "transform_id": "brightness_contrast",
            "parameters": dict(specs["brightness_contrast"]["parameters"]),
            "preset": specs["brightness_contrast"]["preset"],
        }

    if "rotation_90" in specs:
        specs["image_variation_2"] = {
            "image": specs["rotation_90"]["image"],
            "transform_id": "rotation",
            "parameters": dict(specs["rotation_90"]["parameters"]),
            "preset": specs["rotation_90"]["preset"],
        }
    elif "grayscale" in specs:
        specs["image_variation_2"] = {
            "image": specs["grayscale"]["image"],
            "transform_id": "grayscale",
            "parameters": dict(specs["grayscale"]["parameters"]),
            "preset": specs["grayscale"]["preset"],
        }
    elif "jpeg_recompress" in specs:
        specs["image_variation_2"] = {
            "image": specs["jpeg_recompress"]["image"],
            "transform_id": "jpeg_recompress",
            "parameters": dict(specs["jpeg_recompress"]["parameters"]),
            "preset": specs["jpeg_recompress"]["preset"],
        }

    return specs


def generate_image_variants(
    image: Image.Image,
    config: ImageVariationConfig | None = None,
) -> Dict[str, Image.Image]:
    """Generate safe label-preserving photometric image variants.

    Returns:
        Dict containing canonical keys:
        ``original``, ``image_variation_1``, ``image_variation_2`` and transform IDs.
    """
    specs = generate_image_variant_specs(image, config=config)
    ids = [k for k in specs.keys() if k != "original"]
    first = ids[0] if len(ids) >= 1 else "original"
    second = ids[1] if len(ids) >= 2 else first

    return {
        "original": specs["original"]["image"],
        "image_variation_1": specs.get("image_variation_1", specs[first])["image"],
        "image_variation_2": specs.get("image_variation_2", specs[second])["image"],
    }


def blur_or_noise_placeholder(image: Image.Image) -> Image.Image:
    """Placeholder for future blur/noise augmentation (unused by default pipeline)."""
    return image


