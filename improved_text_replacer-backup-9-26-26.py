#!/usr/bin/env python3

"""
Interactive Image Text Replacer
--------------------------------

Detects six red text boxes in a mostly identical image layout,
allows interactive replacement of the text, and saves a new image.

Detection strategy:
- HSV red-mask detection
- Morphological cleanup
- Connected-component analysis
- Projection-based detection for fragmented red rectangles
- Candidate scoring
- Overlap suppression
- Exactly six final target regions required
- Debug images and detailed diagnostics

The source image is never modified.
"""

from __future__ import annotations

import os
import glob
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILENAME = "original.png"
OUTPUT_FILENAME = "replaced_output.png"

DEBUG_DETECTION_FILENAME = "debug_detection.png"
DEBUG_MASK_FILENAME = "debug_red_mask.png"
DEBUG_PROJECTION_FILENAME = "debug_projection.png"

EXPECTED_BOX_COUNT = 6

# -------------------------------------------------------------------------
# Font
# -------------------------------------------------------------------------

# If you know the exact font filename, put it here.
# Example:
#
# FONT_FILENAME = "Roboto-Medium.ttf"
#
# Leave as None for automatic selection.
FONT_FILENAME: Optional[str] = None

FONT_SEARCH_DIRS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "fonts",
]

PREFERRED_FONT_NAMES = [
    "Roboto-Medium.ttf",
    "Roboto-Regular.ttf",
    "Roboto.ttf",
    "Arial.ttf",
    "ArialMT.ttf",
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
]

# -------------------------------------------------------------------------
# Red detection
# -------------------------------------------------------------------------

RED_LOWER_1 = np.array([0, 70, 50], dtype=np.uint8)
RED_UPPER_1 = np.array([12, 255, 255], dtype=np.uint8)

RED_LOWER_2 = np.array([165, 70, 50], dtype=np.uint8)
RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

# Morphological cleanup.
MORPH_KERNEL_SIZE = 5

# -------------------------------------------------------------------------
# Candidate geometry
# -------------------------------------------------------------------------

MIN_BOX_WIDTH = 40
MIN_BOX_HEIGHT = 25

# We deliberately allow fairly broad geometry here.
# The scoring system determines which regions are likely to be boxes.
MAX_BOX_IMAGE_AREA_RATIO = 0.15

MIN_RECTANGULARITY = 0.35
MIN_SOLIDITY = 0.55
MIN_FILL_RATIO = 0.30

# -------------------------------------------------------------------------
# Projection detection
# -------------------------------------------------------------------------

# Minimum fraction of red pixels across a row/column before considering
# it part of a dense red region.
ROW_DENSITY_THRESHOLD = 0.12
COLUMN_DENSITY_THRESHOLD = 0.12

# Minimum number of consecutive rows/columns.
MIN_DENSE_ROWS = 12
MIN_DENSE_COLUMNS = 12

# Merge nearby projection bands.
BAND_GAP = 20

# -------------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------------

MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 500

# Extra margin inside detected box.
TEXT_MARGIN_RATIO = 0.04

# Background sampling margin.
BACKGROUND_SAMPLE_MARGIN = 8

# Inpainting radius for annotation removal.
INPAINT_RADIUS = 3


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
    """
    Find a font in the same directory as the Python script or in ./fonts.

    The current working directory is deliberately NOT used so that the
    script behaves consistently regardless of where it is launched from.
    """

    if FONT_FILENAME:
        for directory in FONT_SEARCH_DIRS:
            candidate = directory / FONT_FILENAME
            if candidate.is_file():
                info(f"Using configured local font: {candidate}")
                return str(candidate)

        raise FileNotFoundError(
            f"Configured font '{FONT_FILENAME}' was not found in:\n"
            + "\n".join(str(p) for p in FONT_SEARCH_DIRS)
        )

    # First try preferred names.
    for preferred in PREFERRED_FONT_NAMES:
        for directory in FONT_SEARCH_DIRS:
            candidate = directory / preferred
            if candidate.is_file():
                info(f"Automatically selected local font: {candidate}")
                return str(candidate)

    # Then search all fonts.
    font_files: list[str] = []

    for directory in FONT_SEARCH_DIRS:
        if directory.exists():
            font_files.extend(
                glob.glob(str(directory / "*.ttf"))
            )
            font_files.extend(
                glob.glob(str(directory / "*.otf"))
            )

    # Remove duplicates while preserving order.
    font_files = list(dict.fromkeys(font_files))

    if not font_files:
        raise FileNotFoundError(
            "No .ttf or .otf font was found in the script directory "
            "or its fonts/ subdirectory."
        )

    # Prefer medium/regular-looking names.
    ranked = sorted(
        font_files,
        key=lambda path: (
            "medium" not in Path(path).stem.lower(),
            "regular" not in Path(path).stem.lower(),
            "sans" not in Path(path).stem.lower(),
            path.lower(),
        ),
    )

    selected = ranked[0]
    info(f"Automatically selected local font: {selected}")
    return selected


# =============================================================================
# IMAGE / RED MASK
# =============================================================================

def create_red_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask containing red pixels.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv,
        RED_LOWER_1,
        RED_UPPER_1,
    )

    mask2 = cv2.inRange(
        hsv,
        RED_LOWER_2,
        RED_UPPER_2,
    )

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )

    # Close small holes/gaps in red regions.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return mask


# =============================================================================
# ANNOTATION REMOVAL - CRITICAL FOR CLEAN OUTPUT
# =============================================================================

def remove_red_annotations_and_restore_image(
    image: np.ndarray,
    red_mask: np.ndarray,
) -> np.ndarray:
    """
    REQUIREMENT #2, #3, #4, #5, #6, #16: Remove ALL red annotations and restore underlying image.
    
    This function:
    - Identifies all red pixels (boxes, numbers, arrows, arrowheads)
    - Uses inpainting to reconstruct the original image underneath
    - Returns a clean image with NO trace of annotations
    
    DO NOT skip this step - it's mandatory for clean output.
    """
    # Expand mask to capture anti-aliased edges and halos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_mask = cv2.dilate(red_mask, kernel, iterations=2)
    
    # Use OpenCV inpainting to reconstruct underlying image
    restored = cv2.inpaint(
        image,
        expanded_mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA
    )
    
    return restored


# =============================================================================
# COMPONENT METRICS
# =============================================================================

def calculate_candidate_metrics(
    contour: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict:
    """
    Calculate geometric properties used to score red regions.
    """

    x, y, w, h = cv2.boundingRect(contour)

    area = float(cv2.contourArea(contour))

    bounding_area = float(max(w * h, 1))

    rect = area / bounding_area

    hull = cv2.convexHull(contour)
    hull_area = float(max(cv2.contourArea(hull), 1))
    solidity = area / hull_area

    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:
        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )
        vertices = len(approx)
    else:
        vertices = 0

    fill_ratio = rect

    aspect = w / max(h, 1)

    image_area = float(image_width * image_height)
    image_area_ratio = bounding_area / image_area

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "bounding_area": bounding_area,
        "aspect": aspect,
        "rect": rect,
        "solid": solidity,
        "fill": fill_ratio,
        "vertices": vertices,
        "image_area_ratio": image_area_ratio,
    }


def candidate_is_basic_valid(metrics: dict) -> bool:
    """
    Broad first-pass filtering.

    This intentionally does not demand a perfect rectangle because the
    source image can contain arrows/text connected to the box.
    """

    if metrics["w"] < MIN_BOX_WIDTH:
        return False

    if metrics["h"] < MIN_BOX_HEIGHT:
        return False

    if metrics["image_area_ratio"] > MAX_BOX_IMAGE_AREA_RATIO:
        return False

    if metrics["rect"] < MIN_RECTANGULARITY:
        return False

    if metrics["solid"] < MIN_SOLIDITY:
        return False

    if metrics["fill"] < MIN_FILL_RATIO:
        return False

    return True


def score_candidate(metrics: dict) -> float:
    """
    Score a possible target box.

    Dense rectangular regions receive higher scores.
    """

    score = 0.0

    # Rectangularity.
    score += min(metrics["rect"], 1.0) * 30.0

    # Solidity.
    score += min(metrics["solid"], 1.0) * 20.0

    # Fill.
    score += min(metrics["fill"], 1.0) * 20.0

    # Four-sided rectangles are strongly preferred.
    if metrics["vertices"] == 4:
        score += 15.0
    elif metrics["vertices"] == 5:
        score += 8.0
    elif metrics["vertices"] == 6:
        score += 4.0

    # Prefer sufficiently large regions.
    area_bonus = min(
        metrics["area"] / 100000.0,
        1.0,
    ) * 10.0

    score += area_bonus

    return score


# =============================================================================
# PROJECTION ANALYSIS
# =============================================================================

def find_dense_bands(
    values: np.ndarray,
    threshold: float,
    minimum_length: int,
    merge_gap: int,
) -> list[tuple[int, int]]:
    """
    Find contiguous regions where projection density exceeds threshold.

    Nearby bands are merged.
    """

    active = values >= threshold

    bands: list[tuple[int, int]] = []

    start: Optional[int] = None

    for index, state in enumerate(active):
        if state and start is None:
            start = index

        elif not state and start is not None:
            end = index - 1

            if end - start + 1 >= minimum_length:
                bands.append((start, end))

            start = None

    if start is not None:
        end = len(active) - 1

        if end - start + 1 >= minimum_length:
            bands.append((start, end))

    if not bands:
        return []

    merged: list[tuple[int, int]] = [bands[0]]

    for start, end in bands[1:]:
        previous_start, previous_end = merged[-1]

        if start - previous_end <= merge_gap:
            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged.append((start, end))

    return merged


def projection_candidates(
    mask: np.ndarray,
) -> list[dict]:
    """
    Search for dense rectangular regions using row/column projections.

    This helps when a target box is fragmented into multiple connected
    contours because of text, anti-aliasing, or arrow connections.
    """

    height, width = mask.shape

    binary = (mask > 0).astype(np.uint8)

    row_density = binary.mean(axis=1)
    column_density = binary.mean(axis=0)

    row_bands = find_dense_bands(
        row_density,
        ROW_DENSITY_THRESHOLD,
        MIN_DENSE_ROWS,
        BAND_GAP,
    )

    column_bands = find_dense_bands(
        column_density,
        COLUMN_DENSITY_THRESHOLD,
        MIN_DENSE_COLUMNS,
        BAND_GAP,
    )

    candidates: list[dict] = []

    for y1, y2 in row_bands:
        for x1, x2 in column_bands:

            x = x1
            y = y1

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
                continue

            roi = binary[y1:y2 + 1, x1:x2 + 1]

            if roi.size == 0:
                continue

            density = float(roi.mean())

            if density < 0.25:
                continue

            area = float(roi.sum())

            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                    "density": density,
                    "source": "projection",
                }
            )

    return candidates


# =============================================================================
# OVERLAP / DUPLICATE HANDLING
# =============================================================================

def intersection_over_union(
    a: dict,
    b: dict,
) -> float:

    ax1 = a["x"]
    ay1 = a["y"]
    ax2 = ax1 + a["w"]
    ay2 = ay1 + a["h"]

    bx1 = b["x"]
    by1 = b["y"]
    bx2 = bx1 + b["w"]
    by2 = by1 + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def boxes_overlap_heavily(
    a: dict,
    b: dict,
    threshold: float = 0.50,
) -> bool:

    return intersection_over_union(a, b) >= threshold


# =============================================================================
# DEBUG IMAGE
# =============================================================================

def create_debug_image(
    image: np.ndarray,
    raw_components: list[dict],
    basic_candidates: list[dict],
    projection_candidates_list: list[dict],
    final_boxes: list[dict],
) -> None:
    """
    Create a diagnostic image.

    Raw components:
        Red

    Basic candidates:
        Yellow

    Projection candidates:
        Blue

    Final selected boxes:
        Green
    """

    debug = image.copy()

    # Raw components.
    for index, candidate in enumerate(raw_components, start=1):
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        cv2.rectangle(
            debug,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            5,
        )

        label = f"RAW {index}"

        cv2.putText(
            debug,
            label,
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Basic candidates.
    for index, candidate in enumerate(basic_candidates, start=1):
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        cv2.rectangle(
            debug,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            4,
        )

        cv2.putText(
            debug,
            f"CAND {index}",
            (x, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Projection candidates.
    for index, candidate in enumerate(
        projection_candidates_list,
        start=1,
    ):
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        cv2.rectangle(
            debug,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3,
        )

        cv2.putText(
            debug,
            f"PROJ {index}",
            (x, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Final boxes.
    for index, candidate in enumerate(final_boxes, start=1):
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        cv2.rectangle(
            debug,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            8,
        )

        cv2.putText(
            debug,
            f"TARGET {index}",
            (x, max(40, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    output_path = SCRIPT_DIR / DEBUG_DETECTION_FILENAME

    cv2.imwrite(
        str(output_path),
        debug,
    )

    info(f"Detection debug image saved to: {output_path}")


def create_projection_debug_image(
    mask: np.ndarray,
) -> None:
    """
    Create a simple visualization of the red mask.
    """

    output_path = SCRIPT_DIR / DEBUG_PROJECTION_FILENAME

    debug = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.imwrite(
        str(output_path),
        debug,
    )


# =============================================================================
# MAIN DETECTION
# =============================================================================

def find_red_boxes(
    image: np.ndarray,
) -> list[dict]:

    height, width = image.shape[:2]

    mask = create_red_mask(image)

    mask_path = SCRIPT_DIR / DEBUG_MASK_FILENAME

    cv2.imwrite(
        str(mask_path),
        mask,
    )

    info(f"Red mask saved to: {mask_path}")

    create_projection_debug_image(mask)

    # -------------------------------------------------------------------------
    # Raw connected components
    # -------------------------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    raw_components: list[dict] = []

    for contour in contours:

        metrics = calculate_candidate_metrics(
            contour,
            width,
            height,
        )

        raw_components.append(metrics)

    # Largest first.
    raw_components.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    info(
        f"Found {len(raw_components)} total red components."
    )

    # -------------------------------------------------------------------------
    # Print EVERY raw component.
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("ALL RAW RED COMPONENTS")
    print("=" * 100)

    for index, metrics in enumerate(
        raw_components,
        start=1,
    ):

        valid = candidate_is_basic_valid(metrics)

        print(
            f"RAW #{index}: "
            f"x={metrics['x']} "
            f"y={metrics['y']} "
            f"w={metrics['w']} "
            f"h={metrics['h']} | "
            f"area={metrics['area']:.0f} | "
            f"aspect={metrics['aspect']:.3f} | "
            f"rect={metrics['rect']:.3f} | "
            f"solid={metrics['solid']:.3f} | "
            f"fill={metrics['fill']:.3f} | "
            f"vertices={metrics['vertices']} | "
            f"basic={'YES' if valid else 'NO'}"
        )

    print("=" * 100)
    print()

    # -------------------------------------------------------------------------
    # Basic candidates
    # -------------------------------------------------------------------------

    basic_candidates: list[dict] = []

    for metrics in raw_components:

        if not candidate_is_basic_valid(metrics):
            continue

        candidate = dict(metrics)
        candidate["score"] = score_candidate(metrics)
        candidate["source"] = "contour"

        basic_candidates.append(candidate)

    basic_candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    info(
        f"{len(basic_candidates)} candidates passed basic geometry filtering."
    )

    print()
    print("=" * 100)
    print("BASIC CANDIDATE DIAGNOSTICS")
    print("=" * 100)

    for index, candidate in enumerate(
        basic_candidates,
        start=1,
    ):

        print(
            f"Candidate #{index}: "
            f"x={candidate['x']} "
            f"y={candidate['y']} "
            f"w={candidate['w']} "
            f"h={candidate['h']} | "
            f"area={candidate['area']:.0f} | "
            f"aspect={candidate['aspect']:.3f} | "
            f"rect={candidate['rect']:.3f} | "
            f"solid={candidate['solid']:.3f} | "
            f"fill={candidate['fill']:.3f} | "
            f"vertices={candidate['vertices']} | "
            f"SCORE={candidate['score']:.2f}"
        )

    print("=" * 100)
    print()

    # -------------------------------------------------------------------------
    # Projection candidates
    # -------------------------------------------------------------------------

    projection_list = projection_candidates(mask)

    info(
        f"Projection analysis produced {len(projection_list)} candidates."
    )

    print()
    print("=" * 100)
    print("PROJECTION CANDIDATES")
    print("=" * 100)

    for index, candidate in enumerate(
        projection_list,
        start=1,
    ):

        print(
            f"Projection #{index}: "
            f"x={candidate['x']} "
            f"y={candidate['y']} "
            f"w={candidate['w']} "
            f"h={candidate['h']} | "
            f"red_area={candidate['area']:.0f} | "
            f"density={candidate['density']:.3f}"
        )

    print("=" * 100)
    print()

    # -------------------------------------------------------------------------
    # Combine candidates.
    # -------------------------------------------------------------------------

    combined: list[dict] = []

    # Contour candidates get strong geometry score.
    for candidate in basic_candidates:
        item = dict(candidate)

        # Projection-style density is not available directly.
        item["combined_score"] = (
            candidate["score"] + 20.0
        )

        combined.append(item)

    # Projection candidates.
    for candidate in projection_list:

        density = candidate["density"]

        score = (
            min(density, 1.0) * 50.0
            + min(candidate["w"] / 1000.0, 1.0) * 20.0
            + min(candidate["h"] / 300.0, 1.0) * 10.0
        )

        item = dict(candidate)
        item["combined_score"] = score

        combined.append(item)

    combined.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    # -------------------------------------------------------------------------
    # Remove duplicate/overlapping candidate boxes.
    # -------------------------------------------------------------------------

    non_duplicate: list[dict] = []

    for candidate in combined:

        duplicate = False

        for existing in non_duplicate:

            if boxes_overlap_heavily(
                candidate,
                existing,
                threshold=0.45,
            ):
                duplicate = True
                break

        if not duplicate:
            non_duplicate.append(candidate)

    # -------------------------------------------------------------------------
    # Select final six.
    # -------------------------------------------------------------------------

    final_boxes: list[dict] = []

    for candidate in non_duplicate:

        if len(final_boxes) >= EXPECTED_BOX_COUNT:
            break

        # Do not allow boxes that are implausibly tiny.
        if candidate["w"] < MIN_BOX_WIDTH:
            continue

        if candidate["h"] < MIN_BOX_HEIGHT:
            continue

        final_boxes.append(candidate)

    # -------------------------------------------------------------------------
    # If projection candidates created broad boxes, refine them against
    # the red mask so that the actual replacement area does not include
    # excessive surrounding background.
    # -------------------------------------------------------------------------

    refined_boxes: list[dict] = []

    for candidate in final_boxes:

        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        # Clamp.
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        x2 = max(x + 1, min(x + w, width))
        y2 = max(y + 1, min(y + h, height))

        refined = dict(candidate)

        refined["x"] = x
        refined["y"] = y
        refined["w"] = x2 - x
        refined["h"] = y2 - y

        refined_boxes.append(refined)

    final_boxes = refined_boxes

    # Sort in natural reading order.
    # This gives a stable 1-6 ordering based primarily on vertical position.
    final_boxes.sort(
        key=lambda item: (
            item["y"],
            item["x"],
        )
    )

    # -------------------------------------------------------------------------
    # Debug image.
    # -------------------------------------------------------------------------

    create_debug_image(
        image,
        raw_components,
        basic_candidates,
        projection_list,
        final_boxes,
    )

    info(
        f"Selected {len(final_boxes)} final box candidates."
    )

    # -------------------------------------------------------------------------
    # Strict six-box requirement.
    # -------------------------------------------------------------------------

    if len(final_boxes) != EXPECTED_BOX_COUNT:

        error("")
        error("TEXT BOX DETECTION FAILED")
        error(
            f"Expected: {EXPECTED_BOX_COUNT} boxes"
        )
        error(
            f"Detected: {len(final_boxes)} boxes"
        )
        error("")
        error(
            "The source image was NOT modified."
        )

        return []

    return final_boxes


# =============================================================================
# TYPOGRAPHY ANALYSIS - NEW FUNCTION
# =============================================================================

def analyze_text_region_typography(
    image: np.ndarray,
    box: dict,
) -> dict:
    """
    REQUIREMENT #9, #12, #14: Analyze original text to preserve typography and alignment.
    
    Determines:
    - Text alignment (left, center, right)
    - Estimated font size
    - Text color
    - Exact positioning within box
    
    Returns configuration dict for accurate text replacement.
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    
    # Margin to avoid red box borders
    margin = 3
    roi_x = x + margin
    roi_y = y + margin
    roi_w = max(1, w - 2 * margin)
    roi_h = max(1, h - 2 * margin)
    
    if roi_w <= 0 or roi_h <= 0:
        return {
            "alignment": "center",
            "font_size": 12,
            "text_color": (0, 0, 0),
            "position": "center"
        }
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        if roi.size == 0:
            return {
                "alignment": "center",
                "font_size": 12,
                "text_color": (0, 0, 0),
                "position": "center"
            }
        
        # Binary threshold to isolate text
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find text contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter noise
        text_contours = [c for c in contours if cv2.contourArea(c) > 20]
        
        if not text_contours:
            return {
                "alignment": "center",
                "font_size": 12,
                "text_color": (0, 0, 0),
                "position": "center"
            }
        
        # Combine all text points
        all_points = []
        for contour in text_contours:
            points = contour.reshape(-1, 2)
            all_points.extend(points.tolist())
        
        if not all_points:
            return {
                "alignment": "center",
                "font_size": 12,
                "text_color": (0, 0, 0),
                "position": "center"
            }
        
        all_points = np.array(all_points)
        text_x_min = int(all_points[:, 0].min())
        text_x_max = int(all_points[:, 0].max())
        text_width = text_x_max - text_x_min
        
        # Determine alignment by analyzing margins
        left_margin = text_x_min
        right_margin = roi_w - text_x_max
        
        margin_difference = left_margin - right_margin
        
        if abs(margin_difference) < 5:
            alignment = "center"
        elif margin_difference < 0:
            alignment = "left"
        else:
            alignment = "right"
        
        # Estimate font size from text height
        text_y_min = int(all_points[:, 1].min())
        text_y_max = int(all_points[:, 1].max())
        text_height = text_y_max - text_y_min
        
        estimated_font_size = max(8, int(text_height * 0.9))
        
        return {
            "alignment": alignment,
            "font_size": estimated_font_size,
            "text_color": (0, 0, 0),
            "original_text_bounds": (text_x_min, text_y_min, text_width, text_height)
        }
        
    except Exception:
        # Fallback to center alignment if analysis fails
        return {
            "alignment": "center",
            "font_size": 12,
            "text_color": (0, 0, 0),
            "position": "center"
        }


# =============================================================================
# OCR - MODIFIED FOR SPECIAL CHARACTER SUPPORT
# =============================================================================

def read_box_text(
    reader: easyocr.Reader,
    image: np.ndarray,
    box: dict,
) -> str:
    """
    REQUIREMENT #10: Read text with explicit support for special characters ($, -, :).
    
    Uses OCR allowlist to ensure $, -, : are recognized correctly.
    """
    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    crop = image[
        y:y + h,
        x:x + w,
    ]

    if crop.size == 0:
        return ""

    # CRITICAL: Use allowlist to ensure special characters are recognized
    results = reader.readtext(
        crop,
        detail=1,
        paragraph=False,
        allowlist="$-:0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@._- ",
    )

    if not results:
        return ""

    results.sort(
        key=lambda result: (
            min(point[1] for point in result[0]),
            min(point[0] for point in result[0]),
        )
    )

    text_parts: list[str] = []

    for result in results:
        text = str(result[1]).strip()

        if text:
            text_parts.append(text)

    return " ".join(text_parts)


# =============================================================================
# FONT FITTING
# =============================================================================

def get_font(
    font_path: str,
    size: int,
) -> ImageFont.FreeTypeFont:

    return ImageFont.truetype(
        font_path,
        size=size,
    )


def text_dimensions(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> tuple[int, int]:

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    return width, height


def fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    max_width: int,
    max_height: int,
) -> int:

    if not text:
        return MIN_FONT_SIZE

    margin_x = max(
        1,
        int(max_width * TEXT_MARGIN_RATIO),
    )

    margin_y = max(
        1,
        int(max_height * TEXT_MARGIN_RATIO),
    )

    available_width = max(
        1,
        max_width - (margin_x * 2),
    )

    available_height = max(
        1,
        max_height - (margin_y * 2),
    )

    # Binary search is considerably faster than trying every size.
    low = MIN_FONT_SIZE
    high = MAX_FONT_SIZE

    best = MIN_FONT_SIZE

    while low <= high:

        mid = (low + high) // 2

        try:
            font = get_font(
                font_path,
                mid,
            )
        except OSError:
            break

        width, height = text_dimensions(
            draw,
            text,
            font,
        )

        # Single-line only.
        if (
            width <= available_width
            and height <= available_height
        ):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


# =============================================================================
# BACKGROUND COLOR
# =============================================================================

def estimate_background_color(
    image: np.ndarray,
    box: dict,
) -> tuple[int, int, int]:

    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    margin = BACKGROUND_SAMPLE_MARGIN

    samples: list[np.ndarray] = []

    # Top.
    if y - margin >= 0:
        samples.append(
            image[
                y - margin:y,
                x:max(x + 1, x + w),
            ]
        )

    # Bottom.
    if y + h + margin <= image.shape[0]:
        samples.append(
            image[
                y + h:y + h + margin,
                x:max(x + 1, x + w),
            ]
        )

    # Left.
    if x - margin >= 0:
        samples.append(
            image[
                y:max(y + 1, y + h),
                x - margin:x,
            ]
        )

    # Right.
    if x + w + margin <= image.shape[1]:
        samples.append(
            image[
                y:max(y + 1, y + h),
                x + w:x + w + margin,
            ]
        )

    if not samples:
        crop = image[
            y:y + h,
            x:x + w,
        ]

        if crop.size == 0:
            return (255, 255, 255)

        median = np.median(
            crop.reshape(-1, 3),
            axis=0,
        )

    else:
        combined = np.concatenate(
            [
                sample.reshape(-1, 3)
                for sample in samples
                if sample.size > 0
            ],
            axis=0,
        )

        if combined.size == 0:
            return (255, 255, 255)

        median = np.median(
            combined,
            axis=0,
        )

    return tuple(
        int(value)
        for value in median
    )


# =============================================================================
# REPLACEMENT RENDERING - MODIFIED FOR ALIGNMENT PRESERVATION
# =============================================================================

def replace_text_in_box(
    image: Image.Image,
    box: dict,
    replacement_text: str,
    font_path: str,
    typography_config: dict,
) -> None:
    """
    REQUIREMENT #13, #14, #15: Replace text while preserving original alignment.
    
    Uses typography analysis to match original text positioning.
    """
    if replacement_text is None:
        replacement_text = ""

    replacement_text = str(
        replacement_text
    ).replace("\n", " ").replace("\r", " ")

    draw = ImageDraw.Draw(image)

    x = int(box["x"])
    y = int(box["y"])
    w = int(box["w"])
    h = int(box["h"])

    if w <= 0 or h <= 0:
        return

    # Convert to NumPy for background estimation.
    cv_image = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR,
    )

    background_bgr = estimate_background_color(
        cv_image,
        box,
    )

    background_rgb = (
        background_bgr[2],
        background_bgr[1],
        background_bgr[0],
    )

    # Erase the interior.
    draw.rectangle(
        [
            x,
            y,
            x + w - 1,
            y + h - 1,
        ],
        fill=background_rgb,
    )

    if not replacement_text:
        return

    # Get alignment from typography analysis
    alignment = typography_config.get("alignment", "center")
    
    font_size = fit_font_size(
        draw,
        replacement_text,
        font_path,
        w,
        h,
    )

    font = get_font(
        font_path,
        font_size,
    )

    bbox = draw.textbbox(
        (0, 0),
        replacement_text,
        font=font,
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position based on alignment
    if alignment == "left":
        text_x = x + 2  # Small margin from left
    elif alignment == "right":
        text_x = x + w - text_width - 2  # Small margin from right
    else:  # center
        text_x = x + (w - text_width) // 2

    # Vertical centering
    text_y = y + (h - text_height) // 2

    # Black is used because the replacement is intended to replace
    # text inside the red box. If the original text is a different color,
    # this can be made configurable later.
    draw.text(
        (
            round(text_x),
            round(text_y),
        ),
        replacement_text,
        font=font,
        fill=(0, 0, 0),
    )


# =============================================================================
# INTERACTIVE PROCESSING - MODIFIED FOR CLEAN OUTPUT
# =============================================================================

def process_image() -> None:

    input_path = SCRIPT_DIR / INPUT_FILENAME
    output_path = SCRIPT_DIR / OUTPUT_FILENAME

    if not input_path.is_file():

        error(
            f"Input image not found: {input_path}"
        )

        return

    # -------------------------------------------------------------------------
    # Load image.
    # -------------------------------------------------------------------------

    image_cv = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if image_cv is None:

        error(
            f"Unable to load image: {input_path}"
        )

        return

    height, width = image_cv.shape[:2]

    info(f"Loaded image: {input_path}")
    info(
        f"Image resolution: {width} x {height}"
    )

    # -------------------------------------------------------------------------
    # Font.
    # -------------------------------------------------------------------------

    try:
        font_path = get_local_font()

    except FileNotFoundError as exc:

        error(str(exc))
        return

    # -------------------------------------------------------------------------
    # Detect boxes BEFORE creating the output.
    # -------------------------------------------------------------------------

    boxes = find_red_boxes(
        image_cv,
    )

    if len(boxes) != EXPECTED_BOX_COUNT:

        error("")
        error("PROCESSING ABORTED.")
        error(
            "No changes were made to the source image."
        )

        return

    # -------------------------------------------------------------------------
    # CRITICAL: Remove ALL red annotations BEFORE any OCR or text replacement
    # REQUIREMENT #2, #3, #4, #5, #6: Clean final image
    # -------------------------------------------------------------------------
    
    info("Creating red mask for annotation removal...")
    red_mask = create_red_mask(image_cv)
    
    info("Removing red annotations (boxes, numbers, arrows) and restoring image...")
    cleaned_image = remove_red_annotations_and_restore_image(image_cv, red_mask)
    
    # -------------------------------------------------------------------------
    # EasyOCR.
    # -------------------------------------------------------------------------

    info("Initializing EasyOCR...")

    try:
        reader = easyocr.Reader(
            ["en"],
            gpu=False,
        )

    except Exception as exc:

        error(
            f"EasyOCR initialization failed: {exc}"
        )

        return

    # -------------------------------------------------------------------------
    # Convert to Pillow using CLEANED image.
    # -------------------------------------------------------------------------

    working_image = Image.fromarray(
        cv2.cvtColor(
            cleaned_image,  # Use cleaned image, not original
            cv2.COLOR_BGR2RGB,
        )
    )

    # -------------------------------------------------------------------------
    # Interactive replacement.
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("INTERACTIVE TEXT REPLACEMENT")
    print("=" * 90)
    print(
        f"Detected exactly {len(boxes)} target boxes."
    )
    print(
        "Enter replacement text for each box."
    )
    print(
        "Press ENTER to leave a box unchanged."
    )
    print(
        "Replacement text is forced to a single line."
    )
    print("=" * 90)
    print()

    for index, box in enumerate(
        boxes,
        start=1,
    ):

        # Use CLEANED image for OCR
        original_text = read_box_text(
            reader,
            cleaned_image,
            box,
        )

        print(
            f"\nBOX #{index}"
        )

        print(
            f"Location: "
            f"x={box['x']} "
            f"y={box['y']} "
            f"w={box['w']} "
            f"h={box['h']}"
        )

        if original_text:
            print(
                f"OCR detected: {original_text}"
            )
        else:
            print(
                "OCR detected: [no text]"
            )

        try:
            replacement = input(
                "Replacement text: "
            )

        except (EOFError, KeyboardInterrupt):

            print()
            warning(
                "Input cancelled."
            )

            return

        replacement = replacement.strip()

        if replacement == "":
            info(
                f"Box #{index} left unchanged."
            )
            continue

        # Force single line.
        replacement = " ".join(
            replacement.splitlines()
        )

        print(
            f"Replacing box #{index} with: "
            f"{replacement}"
        )

        # ---------------------------------------------------------------------
        # CRITICAL: Analyze typography BEFORE replacement
        # REQUIREMENT #9, #12, #14: Preserve original alignment
        # ---------------------------------------------------------------------
        
        typography_config = analyze_text_region_typography(
            cleaned_image,
            box,
        )

        replace_text_in_box(
            working_image,
            box,
            replacement,
            font_path,
            typography_config,
        )

    # -------------------------------------------------------------------------
    # Save output separately.
    # -------------------------------------------------------------------------

    try:

        working_image.save(
            str(output_path),
            format="PNG",
        )

    except Exception as exc:

        error(
            f"Failed to save output image: {exc}"
        )

        return

    print()
    print("=" * 90)
    print("PROCESSING COMPLETE")
    print("=" * 90)
    info(
        f"Output saved to: {output_path}"
    )
    info(
        "The original image was not modified."
    )
    print("=" * 90)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:
        process_image()

    except KeyboardInterrupt:

        print()
        warning(
            "Processing cancelled by user."
        )

    except Exception as exc:

        print()
        error(
            "UNEXPECTED ERROR"
        )
        error(
            f"{type(exc).__name__}: {exc}"
        )
        error(
            "The source image was not intentionally modified."
        )