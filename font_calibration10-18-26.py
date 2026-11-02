#!/usr/bin/env python3
"""Per-box font calibration for improved_text_replacer.py.

Estimates original text ink bounds, foreground color, and the closest local
font/size based on the OCR text's rendered width-to-height ratio. This is an
image-based estimate, not guaranteed recovery of the original font metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font_candidates(script_dir: Path, preferred_font: str | None) -> list[Path]:
    search_dirs = [script_dir, script_dir / "fonts"]
    found: list[Path] = []
    if preferred_font:
        for directory in search_dirs:
            p = directory / preferred_font
            if p.is_file():
                found.append(p)
    for directory in search_dirs:
        if directory.exists():
            found.extend(directory.glob("*.ttf"))
            found.extend(directory.glob("*.otf"))
    unique = list(dict.fromkeys(p.resolve() for p in found if p.is_file()))
    preferred_names = ["Roboto-Medium.ttf", "Roboto-Regular.ttf", "Roboto.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    unique.sort(key=lambda p: (preferred_names.index(p.name) if p.name in preferred_names else 999, p.name.lower()))
    return unique


def _measure_ink(crop_rgb: np.ndarray, background_rgb: tuple[int, int, int]) -> dict[str, Any]:
    """Find non-background, non-red ink pixels in a box crop."""
    pixels = crop_rgb.astype(np.int16)
    bg = np.asarray(background_rgb, dtype=np.int16)
    delta = np.max(np.abs(pixels - bg), axis=2)
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    red = (r > g * 1.35) & (r > b * 1.20) & (r > 70)
    mask = (delta > 35) & (~red)
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return {"found": False, "width": 0, "height": 0, "bbox": None, "color_rgb": [0, 0, 0]}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    ink_pixels = pixels[mask]
    color = np.median(ink_pixels, axis=0).clip(0, 255).astype(int).tolist()
    return {"found": True, "width": x1 - x0, "height": y1 - y0,
            "bbox": [x0, y0, x1, y1], "color_rgb": color}


def calibrate_font_parameters(
    image_rgb: Image.Image,
    boxes: list[dict],
    original_texts: list[str],
    default_font_path: str,
    background_rgb: tuple[int, int, int],
    script_dir: Path,
) -> list[dict[str, Any]]:
    """Run calibration before replacement and return one config per box."""
    fonts = _font_candidates(script_dir, Path(default_font_path).name)
    if not fonts:
        fonts = [Path(default_font_path)]

    image_array = np.asarray(image_rgb)
    draw = ImageDraw.Draw(Image.new("RGB", (2, 2), background_rgb))
    configs: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []

    print("\n" + "=" * 78)
    print("FONT CALIBRATION: measuring original lettering before edits")
    print("=" * 78)

    for index, (box, raw_text) in enumerate(zip(boxes, original_texts), start=1):
        inner = box["inner"]
        x, y, w, h = inner["x"], inner["y"], inner["w"], inner["h"]
        crop = image_array[y:y + h, x:x + w]
        measured = _measure_ink(crop, background_rgb) if crop.size else {
            "found": False, "width": 0, "height": 0, "bbox": None, "color_rgb": [0, 0, 0]
        }
        text = " ".join((raw_text or "").split())
        chosen_font = Path(default_font_path)
        chosen_size = max(8, min(500, int(h * 0.65)))
        best_score = float("inf")
        rendered_ratio = None

        if measured["found"] and text:
            target_ratio = measured["width"] / max(measured["height"], 1)
            target_height = measured["height"]
            # Compare candidate fonts by glyph width/height ratio at a size
            # that approximates the measured ink height.
            for font_path in fonts:
                for size in range(max(8, target_height - 8), min(500, target_height + 18) + 1):
                    try:
                        font = ImageFont.truetype(str(font_path), size=size)
                        bbox = draw.textbbox((0, 0), text, font=font)
                    except (OSError, ValueError):
                        continue
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if tw <= 0 or th <= 0:
                        continue
                    ratio_error = abs((tw / th) - target_ratio) / max(target_ratio, 0.1)
                    height_error = abs(th - target_height) / max(target_height, 1)
                    score = ratio_error * 0.7 + height_error * 0.3
                    if score < best_score:
                        best_score = score
                        chosen_font = font_path
                        chosen_size = size
                        rendered_ratio = round(tw / th, 4)

        config = {
            "box_index": index,
            "font_path": str(chosen_font),
            "font_size": int(chosen_size),
            "text_color_rgb": measured["color_rgb"] if measured["found"] else ([0, 0, 0] if sum(background_rgb) > 380 else [255, 255, 255]),
            "original_ink_bbox_relative": measured["bbox"],
            "original_ink_width_px": measured["width"],
            "original_ink_height_px": measured["height"],
            "original_text": text,
            "font_match_score": round(best_score, 5) if best_score != float("inf") else None,
            "rendered_width_height_ratio": rendered_ratio,
            "calibration_confidence": "estimated" if measured["found"] and text else "fallback",
        }
        configs.append(config)
        report.append(config)
        print(
            f"BOX #{index}: font={chosen_font.name}, size={chosen_size}, "
            f"ink={measured['width']}x{measured['height']} px, "
            f"confidence={config['calibration_confidence']}"
        )

    report_path = script_dir / "font_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibration report saved to: {report_path}")
    print("Note: font identification is an estimate from pixels, not embedded metadata.")
    print("=" * 78 + "\n")
    return configs
