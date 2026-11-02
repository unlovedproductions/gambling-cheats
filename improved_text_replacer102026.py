#!/usr/bin/env python3
"""
Interactive Image Text Replacer
-------------------------------

Detects six outlined red target boxes, asks for replacement text, and saves
a separate output image. The source image is never modified.

Designed for repeated images with:
- A solid-color background
- Six outlined target boxes
- Red box outlines
- Arrow annotations that may touch or point toward boxes
- Replacement text kept on one line

Detection fixes:
- Uses contour hierarchy and bounding-box containment to recognize the inner
  and outer contours of the same outlined rectangle.
- Groups nested contours so one physical box is counted only once.
- Uses the inner contour as the text area when available, preserving the
  outline during replacement.
- Writes debug_red_mask.png and debug_detection.png.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from font_calibration import calibrate_font_parameters


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILENAME = "original.png"
OUTPUT_FILENAME = "replaced_output.png"
DEBUG_DETECTION_FILENAME = "debug_detection.png"
DEBUG_MASK_FILENAME = "debug_red_mask.png"

EXPECTED_BOX_COUNT = 6

# Set this to a filename in the script directory or ./fonts to force a font.
# None means automatically select a local font.
FONT_FILENAME: Optional[str] = None

FONT_SEARCH_DIRS = [SCRIPT_DIR, SCRIPT_DIR / "fonts"]
PREFERRED_FONT_NAMES = [
    "Roboto-Medium.ttf",
    "Roboto-Regular.ttf",
    "Roboto.ttf",
    "Arial.ttf",
    "ArialMT.ttf",
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
]

# HSV red ranges, including reds that wrap around the hue boundary.
RED_LOWER_1 = np.array([0, 25, 35], dtype=np.uint8)
RED_UPPER_1 = np.array([18, 255, 255], dtype=np.uint8)
RED_LOWER_2 = np.array([158, 25, 35], dtype=np.uint8)
RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

MORPH_KERNEL_SIZE = 3

MIN_BOX_WIDTH = 40
MIN_BOX_HEIGHT = 25
MAX_BOX_IMAGE_AREA_RATIO = 0.15
MIN_RECTANGULARITY = 0.35
MIN_SOLIDITY = 0.50

MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 500
TEXT_MARGIN_RATIO = 0.04
BACKGROUND_SAMPLE_MARGIN = 8

# Per-box corrections based on the user's local visual review:
# Box 2 uses bold text and a 12% size boost; box 3 is biased slightly right.
# These are one-based box numbers after sorting targets top-to-bottom.

# Candidate rectangles whose edges fall within this many pixels of one
# another can be considered nested contours of the same outlined box.
NESTED_EDGE_TOLERANCE = 4

# A valid inner contour should be smaller than its outer contour, but still
# occupy a substantial part of the same rectangle.
NESTED_MIN_AREA_RATIO = 0.45
NESTED_MAX_AREA_RATIO = 0.98


# =============================================================================
# LOGGING
# =============================================================================

def info(message: str) -> None:
    print(f"INFO: {message}")


def warning(message: str) -> None:
    print(f"WARNING: {message}")


def error(message: str) -> None:
    print(f"ERROR: {message}")


# =============================================================================
# FONT HANDLING
# =============================================================================

def get_local_font() -> str:
    """Find a font relative to this script, not the current working directory."""
    if FONT_FILENAME:
        for directory in FONT_SEARCH_DIRS:
            candidate = directory / FONT_FILENAME
            if candidate.is_file():
                info(f"Using configured local font: {candidate}")
                return str(candidate)
        raise FileNotFoundError(
            f"Configured font {FONT_FILENAME!r} was not found in "
            f"{', '.join(str(p) for p in FONT_SEARCH_DIRS)}"
        )

    for preferred in PREFERRED_FONT_NAMES:
        for directory in FONT_SEARCH_DIRS:
            candidate = directory / preferred
            if candidate.is_file():
                info(f"Automatically selected local font: {candidate}")
                return str(candidate)

    font_files: list[str] = []
    for directory in FONT_SEARCH_DIRS:
        if directory.exists():
            font_files.extend(glob.glob(str(directory / "*.ttf")))
            font_files.extend(glob.glob(str(directory / "*.otf")))

    font_files = list(dict.fromkeys(font_files))
    if not font_files:
        raise FileNotFoundError(
            "No .ttf or .otf font found beside the script or in its fonts/ folder."
        )

    font_files.sort(
        key=lambda p: (
            "medium" not in Path(p).stem.lower(),
            "regular" not in Path(p).stem.lower(),
            "sans" not in Path(p).stem.lower(),
            p.lower(),
        )
    )
    info(f"Automatically selected local font: {font_files[0]}")
    return font_files[0]


# =============================================================================
# MASK AND GEOMETRY
# =============================================================================

def create_red_mask(image: np.ndarray) -> np.ndarray:
    """Detect red markup plus lighter antialiased pink edges.

    Input is BGR. HSV handles hue variation; the RGB-channel test catches
    low-saturation pink pixels produced when red strokes are antialiased.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    hsv_mask = cv2.bitwise_or(mask1, mask2)

    b, g, r = cv2.split(image)
    red_chroma = (
        (r.astype(np.int16) > g.astype(np.int16) + 14)
        & (r.astype(np.int16) > b.astype(np.int16) + 14)
        & (r > 60)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(hsv_mask, red_chroma)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def contour_metrics(
    contour: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict:
    x, y, w, h = cv2.boundingRect(contour)
    contour_area = float(cv2.contourArea(contour))
    bbox_area = float(max(w * h, 1))
    hull = cv2.convexHull(contour)
    hull_area = float(max(cv2.contourArea(hull), 1))
    perimeter = cv2.arcLength(contour, True)
    approximation = (
        cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if perimeter > 0 else []
    )
    return {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": contour_area,
        "bbox_area": bbox_area,
        "aspect": w / max(h, 1),
        "rect": contour_area / bbox_area,
        "solid": contour_area / hull_area,
        "vertices": len(approximation),
        "image_area_ratio": bbox_area / max(image_width * image_height, 1),
    }


def basic_candidate_valid(m: dict) -> bool:
    if m["w"] < MIN_BOX_WIDTH or m["h"] < MIN_BOX_HEIGHT:
        return False
    if m["image_area_ratio"] > MAX_BOX_IMAGE_AREA_RATIO:
        return False
    if m["rect"] < MIN_RECTANGULARITY or m["solid"] < MIN_SOLIDITY:
        return False
    # Extremely thin arrows and long lines are unlikely to be target boxes.
    if m["aspect"] < 0.45 or m["aspect"] > 12.0:
        return False
    return True


def score_candidate(m: dict) -> float:
    score = 30 * min(m["rect"], 1.0)
    score += 20 * min(m["solid"], 1.0)
    score += 15 if m["vertices"] == 4 else 5
    score += min(m["area"] / 100000.0, 1.0) * 10
    # Give moderate preference to boxes with a plausible outline aspect ratio.
    if 0.65 <= m["aspect"] <= 8.0:
        score += 5
    return score


def box_edges(box: dict) -> tuple[int, int, int, int]:
    return (
        box["x"],
        box["y"],
        box["x"] + box["w"],
        box["y"] + box["h"],
    )


def is_nested_rectangle(outer: dict, inner: dict) -> bool:
    """
    True when inner is geometrically inside outer and its edges resemble the
    inner edge of the same outlined rectangle. This avoids counting a border's
    outer and inner contours as two separate targets.
    """
    ox1, oy1, ox2, oy2 = box_edges(outer)
    ix1, iy1, ix2, iy2 = box_edges(inner)

    if inner["bbox_area"] >= outer["bbox_area"]:
        return False

    ratio = inner["bbox_area"] / max(outer["bbox_area"], 1)
    if not (NESTED_MIN_AREA_RATIO <= ratio <= NESTED_MAX_AREA_RATIO):
        return False

    # Every inner edge must be inside the outer edge, within a reasonable
    # border-thickness range. The tolerance handles small raster differences.
    left_gap = ix1 - ox1
    top_gap = iy1 - oy1
    right_gap = ox2 - ix2
    bottom_gap = oy2 - iy2

    gaps = (left_gap, top_gap, right_gap, bottom_gap)
    if any(gap < -NESTED_EDGE_TOLERANCE for gap in gaps):
        return False
    if any(gap > min(60, max(24, int(min(outer["w"], outer["h"]) * 0.30)))
           for gap in gaps):
        return False

    # The opposing border widths should be reasonably similar.
    if abs(left_gap - right_gap) > max(14, outer["w"] * 0.10):
        return False
    if abs(top_gap - bottom_gap) > max(14, outer["h"] * 0.12):
        return False

    return True


def intersection_over_union(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = box_edges(a)
    bx1, by1, bx2, by2 = box_edges(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a["bbox_area"] + b["bbox_area"] - intersection
    return intersection / union if union > 0 else 0.0


# =============================================================================
# DETECTION AND NESTED-CONTOUR GROUPING
# =============================================================================

def find_red_boxes(image: np.ndarray) -> list[dict]:
    height, width = image.shape[:2]
    mask = create_red_mask(image)

    mask_path = SCRIPT_DIR / DEBUG_MASK_FILENAME
    cv2.imwrite(str(mask_path), mask)
    info(f"Red mask saved to: {mask_path}")

    # RETR_TREE retains parent/child relationships between outer and inner
    # contours. Geometric containment is also checked because antialiasing and
    # arrow contacts can make the hierarchy imperfect.
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    raw: list[dict] = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]

    for idx, contour in enumerate(contours):
        m = contour_metrics(contour, width, height)
        m["contour_index"] = idx
        m["parent"] = int(hierarchy[idx][3]) if hierarchy is not None else -1
        m["basic"] = basic_candidate_valid(m)
        if m["basic"]:
            m["score"] = score_candidate(m)
            raw.append(m)

    raw.sort(key=lambda m: m["bbox_area"], reverse=True)

    info(f"Found {len(contours)} total contours.")
    print("\n" + "=" * 100)
    print("RECTANGLE CANDIDATE DIAGNOSTICS")
    print("=" * 100)

    for index, candidate in enumerate(raw, start=1):
        candidate["nested_children"] = []
        print(
            f"Candidate #{index}: x={candidate['x']} y={candidate['y']} "
            f"w={candidate['w']} h={candidate['h']} "
            f"| area={candidate['area']:.0f} "
            f"| aspect={candidate['aspect']:.3f} "
            f"| rect={candidate['rect']:.3f} "
            f"| solid={candidate['solid']:.3f} "
            f"| vertices={candidate['vertices']} "
            f"| parent={candidate['parent']} "
            f"| score={candidate['score']:.2f}"
        )
    print("=" * 100)

    # Build groups. Each group represents one physical rectangle. A candidate
    # is attached to an existing outer rectangle only if it is genuinely
    # nested with a similar border inset. Merely overlapping boxes are not
    # grouped, so the two intentionally overlapping horizontal boxes survive.
    groups: list[dict] = []
    for candidate in raw:
        matching_group = None
        for group in groups:
            outer = group["outer"]
            if is_nested_rectangle(outer, candidate):
                matching_group = group
                break

        if matching_group is None:
            groups.append({"outer": candidate, "members": [candidate]})
        else:
            matching_group["members"].append(candidate)
            matching_group["outer"]["nested_children"].append(candidate)

    # Prefer the largest bounding rectangle for each group. The smallest
    # nested member is retained as the interior/text area when available.
    physical_boxes: list[dict] = []
    for group in groups:
        members = group["members"]
        outer = max(members, key=lambda m: m["bbox_area"])
        inner_candidates = [
            m for m in members
            if m is not outer and is_nested_rectangle(outer, m)
        ]
        inner = min(inner_candidates, key=lambda m: m["bbox_area"]) if inner_candidates else None

        box = dict(outer)
        box["inner"] = None
        if inner is not None:
            box["inner"] = {
                "x": inner["x"],
                "y": inner["y"],
                "w": inner["w"],
                "h": inner["h"],
            }
        else:
            # Fallback when an inner contour is missing: infer a modest border
            # inset. The detected inner contour is preferred whenever present.
            inset = max(3, min(18, round(min(outer["w"], outer["h"]) * 0.07)))
            box["inner"] = {
                "x": outer["x"] + inset,
                "y": outer["y"] + inset,
                "w": max(1, outer["w"] - 2 * inset),
                "h": max(1, outer["h"] - 2 * inset),
            }
        box["score"] = outer["score"]
        box["group_size"] = len(members)
        physical_boxes.append(box)

    # When more than six non-nested candidates remain, rank them and retain
    # the six strongest rectangle candidates. In the reported case, grouping
    # the two outer/inner pairs should turn eight candidates into six boxes.
    physical_boxes.sort(key=lambda b: b["score"], reverse=True)
    if len(physical_boxes) > EXPECTED_BOX_COUNT:
        warning(
            f"{len(physical_boxes)} distinct candidates remain after nested "
            f"contours were grouped; selecting the {EXPECTED_BOX_COUNT} highest-scoring."
        )
        physical_boxes = physical_boxes[:EXPECTED_BOX_COUNT]

    physical_boxes.sort(key=lambda b: (b["y"], b["x"]))

    # Debug overlay: red = raw candidates, green = selected physical boxes,
    # cyan = detected inner/text region.
    debug = image.copy()
    for index, candidate in enumerate(raw, start=1):
        cv2.rectangle(
            debug,
            (candidate["x"], candidate["y"]),
            (candidate["x"] + candidate["w"] - 1, candidate["y"] + candidate["h"] - 1),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            debug,
            f"C{index}",
            (candidate["x"], max(20, candidate["y"] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for index, box in enumerate(physical_boxes, start=1):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 5)
        inner = box["inner"]
        cv2.rectangle(
            debug,
            (inner["x"], inner["y"]),
            (inner["x"] + inner["w"] - 1, inner["y"] + inner["h"] - 1),
            (255, 255, 0),
            3,
        )
        cv2.putText(
            debug,
            f"TARGET {index} (group={box['group_size']})",
            (x, min(height - 10, y + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    debug_path = SCRIPT_DIR / DEBUG_DETECTION_FILENAME
    cv2.imwrite(str(debug_path), debug)
    info(f"Detection debug image saved to: {debug_path}")
    info(f"Selected {len(physical_boxes)} distinct physical box candidates.")

    if len(physical_boxes) != EXPECTED_BOX_COUNT:
        error("TEXT BOX DETECTION FAILED")
        error(f"Expected: {EXPECTED_BOX_COUNT} boxes")
        error(f"Detected: {len(physical_boxes)} boxes")
        error("The source image was NOT modified.")
        return []

    return physical_boxes


# =============================================================================
# OCR
# =============================================================================

def read_box_text(reader: easyocr.Reader, image: np.ndarray, box: dict) -> str:
    inner = box["inner"]
    x, y, w, h = inner["x"], inner["y"], inner["w"], inner["h"]
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return ""

    results = reader.readtext(crop, detail=1, paragraph=False)
    if not results:
        return ""

    results.sort(
        key=lambda result: (
            min(point[1] for point in result[0]),
            min(point[0] for point in result[0]),
        )
    )
    return " ".join(str(result[1]).strip() for result in results if str(result[1]).strip())


# =============================================================================
# BACKGROUND / FOREGROUND COLOR
# =============================================================================

def estimate_background_color(image_bgr: np.ndarray, box: Optional[dict] = None) -> tuple[int, int, int]:
    """Estimate the image's solid background from the outer border pixels.

    Do not sample immediately beside a target rectangle: those samples can be
    dominated by red borders/arrows, which was the cause of red-filled boxes.
    The user confirmed the background is always one solid color, so the image
    perimeter is the most reliable source.
    """
    height, width = image_bgr.shape[:2]
    strip = max(8, min(40, round(min(height, width) * 0.01)))
    border_parts = [
        image_bgr[:strip, :, :].reshape(-1, 3),
        image_bgr[-strip:, :, :].reshape(-1, 3),
        image_bgr[:, :strip, :].reshape(-1, 3),
        image_bgr[:, -strip:, :].reshape(-1, 3),
    ]
    pixels = np.concatenate(border_parts, axis=0).astype(np.int16)

    # Exclude strong red and antialiased pink markup before estimating background.
    b, g, r = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    red_like = (r > g + 14) & (r > b + 14) & (r > 60)
    clean = pixels[~red_like]
    if len(clean) < max(100, len(pixels) // 10):
        clean = pixels

    # For a solid background, median channel values resist occasional text,
    # edge antialiasing, and annotations that touch the image border.
    median = np.median(clean, axis=0)
    return tuple(int(np.clip(v, 0, 255)) for v in median)


def remove_red_annotations(image: Image.Image, background_rgb: tuple[int, int, int]) -> None:
    """Remove all red annotation pixels (box outlines, arrows, and numbers).

    The target background is solid, so replacing red markup with the global
    background color is more reliable than trying to inpaint each shape.
    A small dilation also catches antialiased red edges.
    """
    rgb = np.asarray(image).copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = create_red_mask(bgr)
    # Expand more aggressively to remove antialiased edges and faint remnants
    # of box outlines. This is safe for this project because the background is
    # confirmed to be a single solid color.
    before = int(np.count_nonzero(mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=1)
    rgb[mask > 0] = np.array(background_rgb, dtype=np.uint8)
    image.paste(Image.fromarray(rgb))

    # Local numerical diagnostics; no image content is exported in this report.
    remaining = int(np.count_nonzero(create_red_mask(
        cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    )))
    info(f"Red cleanup: detected {before} red/pink pixels before edge expansion; "
         f"{remaining} red-like pixels remain after cleanup.")


def estimate_text_color(image_rgb: Image.Image, inner: dict, background_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Estimate the original text color from pixels that differ substantially
    from the solid background. Red outline pixels are excluded.
    """
    arr = np.asarray(image_rgb)
    x, y, w, h = inner["x"], inner["y"], inner["w"], inner["h"]
    crop = arr[y:y + h, x:x + w]
    if crop.size == 0:
        return (0, 0, 0)

    bg = np.array(background_rgb, dtype=np.int16)
    pixels = crop.reshape(-1, 3).astype(np.int16)
    distance = np.max(np.abs(pixels - bg), axis=1)
    selected = pixels[distance > 35]
    if len(selected) == 0:
        # Standard readable fallback for a very clean/empty box.
        return (255, 255, 255) if sum(background_rgb) < 380 else (0, 0, 0)

    # Ignore red pixels if the outline has slightly entered the inner crop.
    is_red = (
        (selected[:, 0] > selected[:, 1] * 1.35)
        & (selected[:, 0] > selected[:, 2] * 1.20)
    )
    selected = selected[~is_red]
    if len(selected) == 0:
        return (255, 255, 255) if sum(background_rgb) < 380 else (0, 0, 0)

    # Median is robust to antialiasing and occasional non-text pixels.
    color = np.median(selected, axis=0)
    return tuple(int(v) for v in color)


# =============================================================================
# TEXT FITTING AND RENDERING
# =============================================================================

def get_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


def fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    max_width: int,
    max_height: int,
    stroke_width: int = 0,
) -> int:
    if not text:
        return MIN_FONT_SIZE

    margin_x = max(1, int(max_width * TEXT_MARGIN_RATIO))
    margin_y = max(1, int(max_height * TEXT_MARGIN_RATIO))
    available_width = max(1, max_width - 2 * margin_x)
    available_height = max(1, max_height - 2 * margin_y)

    low, high = MIN_FONT_SIZE, MAX_FONT_SIZE
    best = MIN_FONT_SIZE
    while low <= high:
        mid = (low + high) // 2
        font = get_font(font_path, mid)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= available_width and th <= available_height:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def replace_text_in_box(
    image: Image.Image,
    box: dict,
    replacement_text: str,
    font_path: str,
    background_rgb: tuple[int, int, int],
    calibration: Optional[dict] = None,
) -> None:
    replacement_text = " ".join(str(replacement_text or "").replace("\r", "\n").splitlines()).strip()
    if not replacement_text:
        return

    draw = ImageDraw.Draw(image)
    inner = box["inner"]
    x, y, w, h = inner["x"], inner["y"], inner["w"], inner["h"]
    if w <= 0 or h <= 0:
        return

    box_index = int((calibration or {}).get("box_index", 0))
    # Keep the erasure strictly inside the box's interior. Box 3 gets an
    # additional left guard because the previous output covered a neighboring
    # character immediately before that box.
    guard_x = max(8, min(22, round(h * 0.07)))
    guard_y = max(3, min(12, round(h * 0.045)))
    if box_index == 3:
        guard_x += max(8, min(22, round(w * 0.025)))
    clean_x0 = x + guard_x
    clean_x1 = x + w - guard_x
    clean_y0 = y + guard_y
    clean_y1 = y + h - guard_y
    if clean_x1 <= clean_x0 or clean_y1 <= clean_y0:
        clean_x0, clean_x1, clean_y0, clean_y1 = x, x + w, y, y + h

    # Only erase the safe interior, not the full detected inner rectangle.
    draw.rectangle((clean_x0, clean_y0, clean_x1 - 1, clean_y1 - 1), fill=background_rgb)
    safe_w, safe_h = clean_x1 - clean_x0, clean_y1 - clean_y0

    calibrated_font_path = (calibration or {}).get("font_path", font_path)
    calibrated_size = int((calibration or {}).get("font_size", 0))
    size_scale = float((calibration or {}).get("font_size_scale", 1.0))
    stroke_width = int((calibration or {}).get("stroke_width", 0))
    target_size = round(calibrated_size * size_scale) if calibrated_size > 0 else 0
    fit_size = fit_font_size(
        draw, replacement_text, calibrated_font_path, safe_w, safe_h, stroke_width
    )
    # Use the calibrated size as a target, but allow the explicit per-box scale
    # to correct known undersizing. Longer replacement strings still shrink to fit.
    font_size = min(target_size, fit_size) if target_size > 0 else fit_size
    font = get_font(calibrated_font_path, font_size)
    bbox = draw.textbbox((0, 0), replacement_text, font=font, stroke_width=stroke_width)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    text_x = clean_x0 + (safe_w - text_width) / 2 - bbox[0]
    # The third target was slightly too far left; bias its text right within
    # the protected interior without changing the other boxes' alignment.
    if box_index == 3:
        text_x += safe_w * 0.025
    text_y = clean_y0 + (safe_h - text_height) / 2 - bbox[1]

    text_color = tuple((calibration or {}).get(
        "text_color_rgb",
        (0, 0, 0) if sum(background_rgb) > 380 else (255, 255, 255),
    ))
    draw.text(
        (round(text_x), round(text_y)), replacement_text, font=font, fill=text_color,
        stroke_width=stroke_width, stroke_fill=text_color,
    )


# =============================================================================
# INTERACTIVE PROCESSING
# =============================================================================

def process_image() -> None:
    input_path = SCRIPT_DIR / INPUT_FILENAME
    output_path = SCRIPT_DIR / OUTPUT_FILENAME

    if not input_path.is_file():
        error(f"Input image not found: {input_path}")
        return

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        error(f"Unable to load image: {input_path}")
        return

    height, width = image_bgr.shape[:2]
    info(f"Loaded image: {input_path}")
    info(f"Image resolution: {width} x {height}")

    try:
        font_path = get_local_font()
    except FileNotFoundError as exc:
        error(str(exc))
        return

    boxes = find_red_boxes(image_bgr)
    if len(boxes) != EXPECTED_BOX_COUNT:
        error("PROCESSING ABORTED. No output image was written.")
        return

    info("Initializing EasyOCR...")
    try:
        reader = easyocr.Reader(["en"], gpu=False)
    except Exception as exc:
        error(f"EasyOCR initialization failed: {exc}")
        return

    original_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    working_image = original_rgb.copy()
    background_bgr = estimate_background_color(image_bgr)
    background_rgb = (background_bgr[2], background_bgr[1], background_bgr[0])
    info(f"Detected solid background RGB: {background_rgb}")

    # Read the original text from every box first, then calibrate font parameters
    # before any pixels are erased or changed.
    original_texts = [read_box_text(reader, image_bgr, box) for box in boxes]
    calibrations = calibrate_font_parameters(
        original_rgb, boxes, original_texts, font_path, background_rgb, SCRIPT_DIR
    )

    print("\n" + "=" * 90)
    print("INTERACTIVE TEXT REPLACEMENT")
    print("=" * 90)
    print(f"Detected exactly {len(boxes)} physical target boxes.")
    print("Enter replacement text for each box. Press ENTER to leave it unchanged.")
    print("Replacement text is kept on a single line.")
    print("Supported punctuation includes: $ : - _ .")
    print("=" * 90)

    for index, box in enumerate(boxes, start=1):
        original_text = original_texts[index - 1]
        calibration = calibrations[index - 1]
        inner = box["inner"]

        print(f"\nBOX #{index}")
        print(
            f"Outer bounds: x={box['x']} y={box['y']} "
            f"w={box['w']} h={box['h']}"
        )
        print(
            f"Text area: x={inner['x']} y={inner['y']} "
            f"w={inner['w']} h={inner['h']}"
        )
        print(f"OCR detected: {original_text or '[no text detected]'}")

        try:
            replacement = input("Replacement text: ")
        except (EOFError, KeyboardInterrupt):
            print()
            warning("Input cancelled. No output image was saved.")
            return

        replacement = " ".join(replacement.replace("\r", "\n").splitlines()).strip()
        if not replacement:
            info(f"Box #{index} left unchanged.")
            continue

        print(f"Replacing box #{index} with: {replacement}")
        replace_text_in_box(working_image, box, replacement, font_path, background_rgb, calibration)

    # Remove the red markup layer from the finished image: box outlines,
    # arrows, numbered markers, and antialiased red edges.
    remove_red_annotations(working_image, background_rgb)
    info("Removed red box outlines, arrows, and numbered markers.")

    try:
        working_image.save(str(output_path), format="PNG")
    except Exception as exc:
        error(f"Failed to save output image: {exc}")
        return

    print("\n" + "=" * 90)
    print("PROCESSING COMPLETE")
    print("=" * 90)
    info(f"Output saved to: {output_path}")
    info("The original image was not modified.")
    print("=" * 90)


if __name__ == "__main__":
    try:
        process_image()
    except KeyboardInterrupt:
        print()
        warning("Processing cancelled by user.")
    except Exception as exc:
        error(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        error("The source image was not intentionally modified.")
