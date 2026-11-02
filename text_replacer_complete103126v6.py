#!/usr/bin/env python3

"""
Interactive Image Text Replacer
--------------------------------

Detects six red text boxes, removes annotations, allows interactive 
replacement of text while preserving alignment, and saves a clean image.

Requirements addressed:
- Detects exactly 6 red boxes (frame-aware: arrows touching a frame and
  stacked frames no longer merge into one region)
- Removes the red annotations (frames, arrows) via inpainting - ONLY those
  pixels; unrelated red image content is preserved
- Replaces text through a tight, glyph-shaped removal mask - never by erasing
  the whole box - and renders with a measured font / size / spacing / baseline
- Every pixel outside the explicitly authorised mask is proven unchanged
  before the output is written
- Supports special characters ($, -, :) in OCR
- Interactive text replacement
- Clean final output without any annotation traces

Module layout
  text_replacer_complete.py  detection, OCR, interactive workflow  (this file)
  text_masks.py              frames, masks, rendering, validation  (engine)
  font_calibration.py        font discovery / ranking / baseline   (typography)

Coordinates: every box dict carries FULL-IMAGE pixel coordinates.
  box["x","y","w","h"]  outer red frame (authorised outer boundary)
  box["inner"]          enclosed interior (where text lives; edits stay inside)
"""

from __future__ import annotations

import os
import glob
import json
import math
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # OCR is only needed for the interactive workflow
    import easyocr
except ImportError:  # pragma: no cover - lets the engine/tests import this module
    easyocr = None

import text_masks as tm
import font_calibration as fc
import font_fetcher as ff


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

FONT_FILENAME: Optional[str] = None

FONT_SEARCH_DIRS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "fonts",
]

# Project folders are searched first; system font folders (per OS) are searched
# for the preferred names and by font_calibration for candidate fonts.
SYSTEM_FONT_DIRS = [d for d in fc.default_font_dirs(SCRIPT_DIR) if d not in FONT_SEARCH_DIRS]

PREFERRED_FONT_NAMES = [
    "Roboto-Medium.ttf",
    "Roboto-Regular.ttf",
    "Roboto.ttf",
    "Arial.ttf",
    "ArialMT.ttf",
    "LiberationSans-Regular.ttf",
    "Arimo-Regular.ttf",
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
]

# -------------------------------------------------------------------------
# Masked replacement / validation
# -------------------------------------------------------------------------

MASK_CONFIG = tm.MaskConfig()

# Boxes known to be bold (1-based, sorted top-to-bottom / left-to-right).  This is
# only a soft prior for font ranking; measured stroke thickness still decides.
BOLD_BOX_INDICES: tuple[int, ...] = (2,)

# If the replacement does not fit: "reject" (default - never shrink/stretch/clip
# automatically) or "shrink" (legacy behaviour: pick the largest size that fits).
ON_OVERFLOW = "shrink"

# In "shrink" mode an overflowing replacement is first slid back inside the box at the
# same size, and only shrunk if it is wider/taller than the box interior.
MASK_CONFIG.allow_clamp = (ON_OVERFLOW == "shrink")

# False (default): ALL red annotation is removed, including numbers placed away from the boxes.
# True: red blobs far from every frame are treated as image content and kept.
PRESERVE_ISOLATED_RED = False
MASK_CONFIG.preserve_isolated_red = PRESERVE_ISOLATED_RED

# Remove isolated white digits (numbers) lying next to a frame. Text with neighbouring letters is never touched.
# Set to False if it ever removes something it should not.
REMOVE_WHITE_NUMBERS = False   # blob-shape guess - replaced by the exact OCR number finder below

# Exact number finder: OCR looks for the digits 1-6 around the frames, each one standing alone,
# and only that glyph is removed. Set FIND_NUMBER_LABELS = False to skip it.
FIND_NUMBER_LABELS = True
NUMBER_LABEL_DIGITS = "123456"
NUMBER_SEARCH_RATIO = 0.10      # search this far (fraction of image width) around each frame
NUMBER_MIN_CONF = 0.25          # OCR digit reads below this are ignored (a lowercase 'o' can read as '5' at ~0.1)
NUMBER_MIN_HEIGHT = 40          # real numbers are ~60 px tall; smaller blobs (letters) are never taken by the size fallback
NUMBER_MAX_HEIGHT = 90
MASK_CONFIG.light_halo_interior_band_px = 5   # white outline cleanup inside a frame never reaches further than this
MASK_CONFIG.remove_light_labels = REMOVE_WHITE_NUMBERS

# Only these font families (substring match on family/file name) are ranked as candidates,
# plus anything you put in ./fonts.  Set to None to rank every installed font.
FONT_FAMILY_ALLOWLIST: Optional[tuple[str, ...]] = (
    "arimo", "liberation sans", "arial", "helvetica", "nimbus sans", "roboto", "open sans", "inter",
    "lato", "noto sans", "source sans", "dejavu sans", "ibm plex sans", "fira sans", "work sans",
    "public sans", "montserrat", "poppins", "nunito sans", "dm sans", "rubik", "barlow", "carlito",
    "segoe", "sf pro", "san francisco", "ubuntu", "cantarell",
)

# Short replacements (1-3 characters) in a roughly square box (monograms / initials) are
# centred in the box interior instead of being anchored to the old ink position.
CENTER_SHORT_TEXT_IN_SQUARE_BOX = True
SHORT_TEXT_MAX_CHARS = 3
# Ask for the original text when OCR finds none (short text such as 1-3 letters is hard
# for OCR); the font ranking compares glyph shapes of this string with the image.
PROMPT_FOR_MISSING_ORIGINAL = True

# Optional manual overrides for the ORIGINAL text of a box (1-based) when OCR is
# wrong.  Font ranking compares glyph shapes of this string against the image.
ORIGINAL_TEXT_OVERRIDES: dict[int, str] = {}

# Optional {box_index: (x0, x1)} horizontal span (relative to the box interior)
# limiting which ink is treated as the target; other ink in the box is kept.
TARGET_SPANS: dict[int, tuple[int, int]] = {}

# Download an open-licensed font when the installed fonts match poorly (needs internet).
#   "ask"  - show what would be downloaded and ask first (default)
#   "auto" - download without asking     "off" - never download
FONT_DOWNLOAD = "ask"
FONT_DOWNLOAD_SCOPE = "core"            # "core" (7 popular sans families) or "all"
FONT_DOWNLOAD_DIR = SCRIPT_DIR / "fonts"
FONT_DOWNLOAD_TIMEOUT = 30.0
POOR_CONFIDENCE = ("fallback", "estimated-low", "ambiguous")

WRITE_DIAGNOSTICS = True
DIAGNOSTICS_DIRNAME = "diagnostics"
REPORT_FILENAME = "replacement_report.json"

# -------------------------------------------------------------------------
# Red detection
# -------------------------------------------------------------------------

RED_LOWER_1 = np.array([0, 70, 50], dtype=np.uint8)
RED_UPPER_1 = np.array([12, 255, 255], dtype=np.uint8)

RED_LOWER_2 = np.array([165, 70, 50], dtype=np.uint8)
RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

MORPH_KERNEL_SIZE = 5

# -------------------------------------------------------------------------
# Candidate geometry
# -------------------------------------------------------------------------

MIN_BOX_WIDTH = 40
MIN_BOX_HEIGHT = 25
MAX_BOX_IMAGE_AREA_RATIO = 0.15
MIN_RECTANGULARITY = 0.35
MIN_SOLIDITY = 0.55
MIN_FILL_RATIO = 0.30

# -------------------------------------------------------------------------
# Projection detection
# -------------------------------------------------------------------------

ROW_DENSITY_THRESHOLD = 0.12
COLUMN_DENSITY_THRESHOLD = 0.12
MIN_DENSE_ROWS = 12
MIN_DENSE_COLUMNS = 12
BAND_GAP = 20

# -------------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------------

MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 500
TEXT_MARGIN_RATIO = 0.04
BACKGROUND_SAMPLE_MARGIN = 8


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
    """Find a font in the script directory or ./fonts subdirectory."""
    
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
            font_files.extend(glob.glob(str(directory / "*.ttf")))
            font_files.extend(glob.glob(str(directory / "*.otf")))

    font_files = list(dict.fromkeys(font_files))

    if not font_files:
        # Nothing in the project folders: look for a preferred font in the
        # system font folders (the choice is logged, never silent).
        for preferred in PREFERRED_FONT_NAMES:
            for directory in SYSTEM_FONT_DIRS:
                if directory.exists():
                    hits = sorted(directory.rglob(preferred))
                    if hits:
                        info(f"No project font found; using system font: {hits[0]}")
                        return str(hits[0])
        raise FileNotFoundError(
            "No .ttf or .otf font was found in the script directory, its fonts/ "
            "subdirectory, or the system font folders."
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
    """Create a binary mask containing red pixels (boxes, numbers, arrows)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


# =============================================================================
# ANNOTATION REMOVAL - CRITICAL FOR CLEAN OUTPUT
# =============================================================================

def remove_red_annotations(image: np.ndarray, red_mask: np.ndarray) -> np.ndarray:
    """
    Remove ALL red annotations and restore underlying image using inpainting.
    
    This removes:
    - Red boxes
    - Red numbers (1, 2, 3, 4, 5, 6)
    - Red arrows and arrowheads
    - Any other red annotation artifacts
    
    Returns a clean image with NO trace of annotations.
    """
    # Expand mask to capture anti-aliased edges and halos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_mask = cv2.dilate(red_mask, kernel, iterations=2)
    
    # Use OpenCV inpainting to reconstruct underlying image
    restored = cv2.inpaint(
        image,
        expanded_mask,
        3,  # inpainting radius
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
    """Calculate geometric properties used to score red regions."""
    
    x, y, w, h = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    bounding_area = float(max(w * h, 1))
    rect = area / bounding_area

    hull = cv2.convexHull(contour)
    hull_area = float(max(cv2.contourArea(hull), 1))
    solidity = area / hull_area

    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
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
    """Broad first-pass filtering."""
    
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
    """Score a possible target box."""
    
    score = 0.0
    score += min(metrics["rect"], 1.0) * 30.0
    score += min(metrics["solid"], 1.0) * 20.0
    score += min(metrics["fill"], 1.0) * 20.0

    if metrics["vertices"] == 4:
        score += 15.0
    elif metrics["vertices"] == 5:
        score += 8.0
    elif metrics["vertices"] == 6:
        score += 4.0

    area_bonus = min(metrics["area"] / 100000.0, 1.0) * 10.0
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
    """Find contiguous regions where projection density exceeds threshold."""
    
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
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))

    return merged


def projection_candidates(mask: np.ndarray) -> list[dict]:
    """Search for dense rectangular regions using row/column projections."""
    
    height, width = mask.shape
    binary = (mask > 0).astype(np.uint8)

    row_density = binary.mean(axis=1)
    column_density = binary.mean(axis=0)

    row_bands = find_dense_bands(
        row_density, ROW_DENSITY_THRESHOLD, MIN_DENSE_ROWS, BAND_GAP
    )
    column_bands = find_dense_bands(
        column_density, COLUMN_DENSITY_THRESHOLD, MIN_DENSE_COLUMNS, BAND_GAP
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

            candidates.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "density": density,
                "source": "projection",
            })

    return candidates


# =============================================================================
# OVERLAP / DUPLICATE HANDLING
# =============================================================================

def intersection_over_union(a: dict, b: dict) -> float:
    """Calculate IoU between two boxes."""
    
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def boxes_overlap_heavily(a: dict, b: dict, threshold: float = 0.50) -> bool:
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
    """Create a diagnostic image showing detection stages."""
    
    debug = image.copy()

    # Raw components - Red
    for index, candidate in enumerate(raw_components, start=1):
        x, y, w, h = candidate["x"], candidate["y"], candidate["w"], candidate["h"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.putText(
            debug, f"RAW {index}", (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
        )

    # Basic candidates - Yellow
    for index, candidate in enumerate(basic_candidates, start=1):
        x, y, w, h = candidate["x"], candidate["y"], candidate["w"], candidate["h"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 4)
        cv2.putText(
            debug, f"CAND {index}", (x, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
        )

    # Projection candidates - Blue
    for index, candidate in enumerate(projection_candidates_list, start=1):
        x, y, w, h = candidate["x"], candidate["y"], candidate["w"], candidate["h"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(
            debug, f"PROJ {index}", (x, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA
        )

    # Final boxes - Green
    for index, candidate in enumerate(final_boxes, start=1):
        x, y, w, h = candidate["x"], candidate["y"], candidate["w"], candidate["h"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 8)
        cv2.putText(
            debug, f"TARGET {index}", (x, max(40, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA
        )
        inner = candidate.get("inner")
        if inner:  # editable interior, drawn thin in cyan
            cv2.rectangle(debug, (inner["x"], inner["y"]),
                          (inner["x"] + inner["w"], inner["y"] + inner["h"]), (255, 255, 0), 2)

    output_path = SCRIPT_DIR / DEBUG_DETECTION_FILENAME
    cv2.imwrite(str(output_path), debug)
    info(f"Detection debug image saved to: {output_path}")


def create_projection_debug_image(mask: np.ndarray) -> None:
    """Create a simple visualization of the red mask."""
    output_path = SCRIPT_DIR / DEBUG_PROJECTION_FILENAME
    debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(output_path), debug)


# =============================================================================
# MAIN DETECTION
# =============================================================================

def find_red_boxes(image: np.ndarray) -> list[dict]:
    """Find exactly 6 red boxes."""
    
    height, width = image.shape[:2]
    mask = create_red_mask(image)

    mask_path = SCRIPT_DIR / DEBUG_MASK_FILENAME
    cv2.imwrite(str(mask_path), mask)
    info(f"Red mask saved to: {mask_path}")

    create_projection_debug_image(mask)

    # Raw connected components
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_components: list[dict] = []

    for contour in contours:
        metrics = calculate_candidate_metrics(contour, width, height)
        raw_components.append(metrics)

    raw_components.sort(key=lambda item: item["area"], reverse=True)
    info(f"Found {len(raw_components)} total red components.")

    # Print raw components
    print()
    print("=" * 100)
    print("ALL RAW RED COMPONENTS")
    print("=" * 100)
    for index, metrics in enumerate(raw_components, start=1):
        valid = candidate_is_basic_valid(metrics)
        print(
            f"RAW #{index}: x={metrics['x']} y={metrics['y']} w={metrics['w']} h={metrics['h']} | "
            f"area={metrics['area']:.0f} | rect={metrics['rect']:.3f} | "
            f"solid={metrics['solid']:.3f} | basic={'YES' if valid else 'NO'}"
        )
    print("=" * 100)
    print()

    # Basic candidates
    basic_candidates: list[dict] = []
    for metrics in raw_components:
        if not candidate_is_basic_valid(metrics):
            continue
        candidate = dict(metrics)
        candidate["score"] = score_candidate(metrics)
        candidate["source"] = "contour"
        basic_candidates.append(candidate)

    basic_candidates.sort(key=lambda item: item["score"], reverse=True)
    info(f"{len(basic_candidates)} candidates passed basic geometry filtering.")

    # Print basic candidates
    print()
    print("=" * 100)
    print("BASIC CANDIDATE DIAGNOSTICS")
    print("=" * 100)
    for index, candidate in enumerate(basic_candidates, start=1):
        print(
            f"Candidate #{index}: x={candidate['x']} y={candidate['y']} w={candidate['w']} h={candidate['h']} | "
            f"SCORE={candidate['score']:.2f}"
        )
    print("=" * 100)
    print()

    # Projection candidates
    projection_list = projection_candidates(mask)
    info(f"Projection analysis produced {len(projection_list)} candidates.")

    print()
    print("=" * 100)
    print("PROJECTION CANDIDATES")
    print("=" * 100)
    for index, candidate in enumerate(projection_list, start=1):
        print(
            f"Projection #{index}: x={candidate['x']} y={candidate['y']} w={candidate['w']} h={candidate['h']} | "
            f"density={candidate['density']:.3f}"
        )
    print("=" * 100)
    print()

    # Combine candidates
    combined: list[dict] = []
    for candidate in basic_candidates:
        item = dict(candidate)
        item["combined_score"] = candidate["score"] + 20.0
        combined.append(item)

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

    combined.sort(key=lambda item: item["combined_score"], reverse=True)

    # Remove duplicates
    non_duplicate: list[dict] = []
    for candidate in combined:
        duplicate = False
        for existing in non_duplicate:
            if boxes_overlap_heavily(candidate, existing, threshold=0.45):
                duplicate = True
                break
        if not duplicate:
            non_duplicate.append(candidate)

    # Select final six
    final_boxes: list[dict] = []
    for candidate in non_duplicate:
        if len(final_boxes) >= EXPECTED_BOX_COUNT:
            break
        if candidate["w"] < MIN_BOX_WIDTH:
            continue
        if candidate["h"] < MIN_BOX_HEIGHT:
            continue
        final_boxes.append(candidate)

    # Refine boxes
    refined_boxes: list[dict] = []
    for candidate in final_boxes:
        x = max(0, min(candidate["x"], width - 1))
        y = max(0, min(candidate["y"], height - 1))
        x2 = max(x + 1, min(x + candidate["w"], width))
        y2 = max(y + 1, min(y + candidate["h"], height))
        
        refined = dict(candidate)
        refined["x"] = x
        refined["y"] = y
        refined["w"] = x2 - x
        refined["h"] = y2 - y
        refined_boxes.append(refined)

    final_boxes = refined_boxes
    final_boxes.sort(key=lambda item: (item["y"], item["x"]))

    info(f"Selected {len(final_boxes)} final box candidates.")

    # ------------------------------------------------------------------
    # Frame-aware reconciliation.  The candidate logic above works from
    # connected red components, so an arrow touching a frame (or two frames
    # sharing an edge) is merged into one oversized region.  Each frame still
    # encloses its own rectangular interior, so the enclosed regions are used
    # to confirm, correct or complete the candidates.  Legacy boxes that agree
    # with a frame are kept unchanged.
    # ------------------------------------------------------------------
    legacy_summary = [(b["x"], b["y"], b["w"], b["h"]) for b in final_boxes]
    frames = tm.detect_frames(mask, MASK_CONFIG)
    notes: list[str] = []
    reconciled = tm.reconcile_boxes(final_boxes, frames, EXPECTED_BOX_COUNT, notes)
    for line in notes:
        info(line)

    # Debug image (shows the final, reconciled boxes)
    create_debug_image(image, raw_components, basic_candidates, projection_list,
                       reconciled if reconciled else final_boxes)

    if len(reconciled) != EXPECTED_BOX_COUNT:
        error("")
        error("TEXT BOX DETECTION FAILED")
        error(f"Expected: {EXPECTED_BOX_COUNT} boxes")
        error(f"Candidate logic: {len(legacy_summary)} boxes; frame detection: {len(frames)} frames")
        error("")
        error("The source image was NOT modified.")
        return []

    changed = [b for b in reconciled if b.get("source") == "frame"]
    if changed:
        info(f"{len(changed)} candidate box(es) were merged with an arrow/neighbour and were "
             "replaced by the enclosed frame region.")
    return reconciled


# =============================================================================
# TYPOGRAPHY ANALYSIS
# =============================================================================

def analyze_text_region(image: np.ndarray, box: dict) -> dict:
    """
    Analyze text region to determine alignment and typography.
    
    Returns dict with:
    - alignment: 'left', 'center', or 'right'
    - font_size: estimated font size
    - text_color: estimated text color (RGB)
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    
    margin = 3
    roi_x = x + margin
    roi_y = y + margin
    roi_w = max(1, w - 2 * margin)
    roi_h = max(1, h - 2 * margin)
    
    if roi_w <= 0 or roi_h <= 0:
        return {"alignment": "center", "font_size": 12, "text_color": (0, 0, 0)}
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        if roi.size == 0:
            return {"alignment": "center", "font_size": 12, "text_color": (0, 0, 0)}
        
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_contours = [c for c in contours if cv2.contourArea(c) > 20]
        
        if not text_contours:
            return {"alignment": "center", "font_size": 12, "text_color": (0, 0, 0)}
        
        all_points = []
        for contour in text_contours:
            points = contour.reshape(-1, 2)
            all_points.extend(points.tolist())
        
        if not all_points:
            return {"alignment": "center", "font_size": 12, "text_color": (0, 0, 0)}
        
        all_points = np.array(all_points)
        text_x_min = int(all_points[:, 0].min())
        text_x_max = int(all_points[:, 0].max())
        text_width = text_x_max - text_x_min
        
        left_margin = text_x_min
        right_margin = roi_w - text_x_max
        difference = left_margin - right_margin
        
        if abs(difference) < 5:
            alignment = "center"
        elif difference < 0:
            alignment = "left"
        else:
            alignment = "right"
        
        text_y_min = int(all_points[:, 1].min())
        text_y_max = int(all_points[:, 1].max())
        text_height = text_y_max - text_y_min
        
        estimated_font_size = max(8, int(text_height * 0.9))
        
        return {
            "alignment": alignment,
            "font_size": estimated_font_size,
            "text_color": (0, 0, 0),
        }
        
    except Exception:
        return {"alignment": "center", "font_size": 12, "text_color": (0, 0, 0)}


# =============================================================================
# OCR WITH SPECIAL CHARACTER SUPPORT
# =============================================================================

def read_box_text(reader: easyocr.Reader, image: np.ndarray, box: dict) -> str:
    """
    Read text from box with explicit support for special characters ($, -, :).
    """
    # Read the enclosed interior (frame already removed), not the outer frame box.
    x, y, w, h = tm.inner_rect_of(box)

    crop = image[y:y + h, x:x + w]
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

    text_parts = []
    for result in results:
        text = str(result[1]).strip()
        if text:
            text_parts.append(text)

    return " ".join(text_parts)


# =============================================================================
# FONT FITTING
# =============================================================================

def get_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


def text_dimensions(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def fit_font_size(draw: ImageDraw.ImageDraw, text: str, font_path: str, max_width: int, max_height: int) -> int:
    """Binary search for best font size."""
    
    if not text:
        return MIN_FONT_SIZE

    margin_x = max(1, int(max_width * TEXT_MARGIN_RATIO))
    margin_y = max(1, int(max_height * TEXT_MARGIN_RATIO))
    available_width = max(1, max_width - (margin_x * 2))
    available_height = max(1, max_height - (margin_y * 2))

    low = MIN_FONT_SIZE
    high = MAX_FONT_SIZE
    best = MIN_FONT_SIZE

    while low <= high:
        mid = (low + high) // 2
        
        try:
            font = get_font(font_path, mid)
        except OSError:
            break

        width, height = text_dimensions(draw, text, font)

        if width <= available_width and height <= available_height:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


# =============================================================================
# BACKGROUND COLOR
# =============================================================================

def estimate_background_color(image: np.ndarray, box: dict) -> tuple[int, int, int]:
    """Estimate background color from surrounding pixels."""
    
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    margin = BACKGROUND_SAMPLE_MARGIN
    samples: list[np.ndarray] = []

    if y - margin >= 0:
        samples.append(image[y - margin:y, x:max(x + 1, x + w)])
    if y + h + margin <= image.shape[0]:
        samples.append(image[y + h:y + h + margin, x:max(x + 1, x + w)])
    if x - margin >= 0:
        samples.append(image[y:max(y + 1, y + h), x - margin:x])
    if x + w + margin <= image.shape[1]:
        samples.append(image[y:max(y + 1, y + h), x + w:x + w + margin])

    if not samples:
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return (255, 255, 255)
        median = np.median(crop.reshape(-1, 3), axis=0)
    else:
        combined = np.concatenate(
            [sample.reshape(-1, 3) for sample in samples if sample.size > 0],
            axis=0,
        )
        if combined.size == 0:
            return (255, 255, 255)
        median = np.median(combined, axis=0)

    return tuple(int(value) for value in median)


# =============================================================================
# REPLACEMENT RENDERING (mask based - never erases the whole box)
# =============================================================================

def _shrink_plan(text: str, an: "tm.InteriorAnalysis", inner_wh: tuple[int, int], typo: dict) -> "tm.ReplacementPlan":
    """Legacy ON_OVERFLOW='shrink': largest font size at which the text fits."""
    size = int(typo["font_size"])
    last: Optional[Exception] = None
    while size >= MIN_FONT_SIZE:
        t = dict(typo)
        t["font_size"] = size
        try:
            plan = tm.plan_replacement(text, an, inner_wh, t, MASK_CONFIG)
            plan.notes.append(f"shrunk to {size}px (ON_OVERFLOW='shrink')")
            return plan
        except tm.PlacementError as exc:
            last = exc
            size -= 1
    raise tm.PlacementError(str(last) if last else "text does not fit at any size")


def replace_text_in_box(
    image: Image.Image,
    box: dict,
    replacement_text: str,
    font_path: str,
    typography: dict,
) -> None:
    """
    Replace text in a box (legacy-compatible, in place on a PIL image).

    Only the original glyph pixels and the new glyph pixels inside the box
    interior are modified.  Raises tm.PlacementError (image untouched) if the
    replacement cannot be placed inside the interior.
    """
    text = " ".join(str(replacement_text or "").split())
    arr = np.array(image.convert("RGB"))
    rect = tm.inner_rect_of(box)
    ix, iy, iw, ih = rect
    an = tm.analyze_interior(arr[iy:iy + ih, ix:ix + iw], MASK_CONFIG)
    if not text:
        if an.found:
            tm.remove_only(arr, rect, an, MASK_CONFIG)
    else:
        typo = dict(typography or {})
        typo.setdefault("font_path", font_path)
        if not typo.get("font_size"):
            typo["font_size"] = max(MIN_FONT_SIZE, int(0.72 * (an.ink_bbox_rel[3] - an.ink_bbox_rel[1]))
                                    if an.ink_bbox_rel else int(ih * 0.5))
        plan = tm.plan_replacement(text, an, (iw, ih), typo, MASK_CONFIG)
        tm.execute_edit(arr, rect, an, plan, MASK_CONFIG)
    image.paste(Image.fromarray(arr))


# =============================================================================
# FONT CALIBRATION + OPTIONAL FONT DOWNLOAD
# =============================================================================

def _calibrate(cleaned_rgb, boxes, original_texts, font_path, analyses, font_dirs) -> list[dict]:
    try:
        return fc.calibrate_font_parameters(
            Image.fromarray(cleaned_rgb), boxes, original_texts, font_path,
            analyses[0].bg_rgb, SCRIPT_DIR,
            analyses=analyses, font_dirs=font_dirs,
            bold_box_indices=BOLD_BOX_INDICES,
            report_path=SCRIPT_DIR / "font_calibration_report.json",
            mask_config=MASK_CONFIG,
            family_allowlist=FONT_FAMILY_ALLOWLIST,
        )
    except Exception as exc:  # never silent: say so and use the default font
        warning(f"Font calibration failed ({type(exc).__name__}: {exc}); using default font {Path(font_path).name}.")
        return [{"font_path": font_path,
                 "font_size": max(MIN_FONT_SIZE, int(0.72 * (a.ink_bbox_rel[3] - a.ink_bbox_rel[1]))
                                  if a.ink_bbox_rel else 20),
                 "calibration_confidence": "fallback"} for a in analyses]


def _is_poor(cfg: dict) -> bool:
    if cfg.get("calibration_confidence") in POOR_CONFIDENCE:
        return True
    return any("poor match" in n for n in cfg.get("notes", []))


def _maybe_download_and_recalibrate(configs, cleaned_rgb, boxes, original_texts, font_path, analyses,
                                    font_dirs) -> list[dict]:
    """If any box matched poorly, offer to download open fonts, then re-rank.

    A box switches to the new calibration only if its match score actually improved;
    otherwise the earlier result stays (the downloaded fonts remain in FONT_DOWNLOAD_DIR).
    """
    poor = [i + 1 for i, c in enumerate(configs) if _is_poor(c) and analyses[i].found and original_texts[i]]
    if not poor or FONT_DOWNLOAD == "off":
        return configs

    installed, _ = fc.discover_fonts(font_dirs)
    todo = ff.missing_entries([f.family for f in installed], FONT_DOWNLOAD_SCOPE)
    print()
    warning(f"Font match is poor/ambiguous for box(es) {poor}.")
    if not todo:
        info("All catalog fonts are already installed; nothing to download.")
        print(ff.PROPRIETARY_NOTE)
        return configs

    print("Open-licensed fonts that are not installed (SIL OFL / Apache 2.0, from github.com/google/fonts):")
    for e in todo:
        print(f"   - {e.family}" + (f"  [{e.note}]" if e.note else ""))
    print(f"They would be saved to: {FONT_DOWNLOAD_DIR}")
    print(ff.PROPRIETARY_NOTE)
    if FONT_DOWNLOAD == "ask":
        try:
            answer = input("Download these fonts and re-run font matching? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return configs
        if answer not in ("y", "yes"):
            info("Font download skipped.")
            return configs

    saved, errors = ff.download_families(todo, FONT_DOWNLOAD_DIR, FONT_DOWNLOAD_TIMEOUT, info)
    for e in errors:
        warning(e)
    if not saved:
        warning("No fonts could be downloaded (offline, blocked or rate limited). Keeping the current font match.")
        return configs

    new_configs = _calibrate(cleaned_rgb, boxes, original_texts, font_path, analyses, font_dirs)
    merged = list(configs)
    for i, (old, new) in enumerate(zip(configs, new_configs)):
        o, n = old.get("font_match_score"), new.get("font_match_score")
        if n is not None and (o is None or n < o - 1e-6):
            merged[i] = new
            info(f"Box #{i + 1}: better match after download: {Path(new['font_path']).name} "
                 f"(score {o} -> {n}, confidence {new.get('calibration_confidence')})")
        else:
            info(f"Box #{i + 1}: downloaded fonts did not improve the match; keeping {Path(old['font_path']).name}")
    return merged


# =============================================================================
# INTERACTIVE PROCESSING
# =============================================================================

def _bbox_abs(bbox_rel: Optional[tuple], rect: tuple[int, int, int, int]) -> Optional[list[int]]:
    if not bbox_rel:
        return None
    ix, iy, _, _ = rect
    return [int(bbox_rel[0]) + ix, int(bbox_rel[1]) + iy, int(bbox_rel[2]) + ix, int(bbox_rel[3]) + iy]



def find_number_labels(reader, image_rgb: np.ndarray, boxes: list[dict], ann_mask: np.ndarray) -> tuple[np.ndarray, list[dict], list[str]]:
    """Locate the annotation numbers (each digit 1-6 once, standing alone) and return a mask of EXACTLY those glyphs.

    OCR (digits only) runs in a band around every frame, never inside a frame interior.  A candidate is accepted
    only if it is a single digit, outside every interior, and has no other ink next to it (so letters of nearby
    text are never taken).  The glyph mask is the pixels inside the digit's box that differ from the local
    background.  Returns (mask, found list, messages).
    """
    H, W = image_rgb.shape[:2]
    out = np.zeros((H, W), dtype=bool)
    found: list[dict] = []
    msgs: list[str] = []
    inner = [tm.inner_rect_of(b) for b in boxes]
    inside_any = np.zeros((H, W), dtype=bool)
    for ix, iy, iw, ih in inner:
        inside_any[iy:iy + ih, ix:ix + iw] = True
    margin = max(30, int(NUMBER_SEARCH_RATIO * W))
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ann_near = cv2.dilate(ann_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    # Local "ink" = pixels that stand out from their surroundings (used for the stand-alone test).
    ink_all = np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 31).astype(np.int16)) > 28
    ink_all &= ~ann_near

    regions = []
    for b in boxes:
        x0 = max(0, int(b["x"]) - margin); y0 = max(0, int(b["y"]) - margin)
        x1 = min(W, int(b["x"] + b["w"]) + margin); y1 = min(H, int(b["y"] + b["h"]) + margin)
        regions.append((x0, y0, x1, y1))

    def treatments(crop):
        """(name, image, scale) - several renderings of the same crop; OCR is tried on each in turn."""
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(up)
        _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        light_on_dark_black = 255 - otsu if float(np.mean(up)) < 128 else otsu
        yield "colour", crop, 1.0
        yield "grey x3", up, 3.0
        yield "inverted x3", 255 - up, 3.0
        yield "contrast x3", clahe, 3.0
        yield "threshold x3", light_on_dark_black, 3.0
        yield "inverted contrast x3", 255 - clahe, 3.0

    cands: list[dict] = []
    seen: set[tuple] = set()
    rejected: list[dict] = []
    best: dict[str, dict] = {}
    passes_used: list[str] = []

    def add_candidates(treatment_idx: int) -> None:
        for (x0, y0, x1, y1) in regions:
            crop = np.ascontiguousarray(bgr[y0:y1, x0:x1])
            tr = list(treatments(crop))
            if treatment_idx >= len(tr):
                continue
            name, img, sc = tr[treatment_idx]
            try:
                res = reader.readtext(img, allowlist=NUMBER_LABEL_DIGITS, detail=1, paragraph=False,
                                      text_threshold=0.25, low_text=0.15, mag_ratio=2.0 if sc == 1.0 else 1.0)
            except Exception as exc:                  # OCR failure must never stop the run
                msgs.append(f"number search ({name}) failed ({type(exc).__name__}: {exc})")
                continue
            for quad, text, conf in res:
                t = str(text).strip()
                if len(t) != 1 or t not in NUMBER_LABEL_DIGITS or float(conf) < NUMBER_MIN_CONF:
                    continue
                xs = [q[0] / sc for q in quad]; ys = [q[1] / sc for q in quad]
                gx0, gy0 = int(min(xs)) + x0, int(min(ys)) + y0
                gx1, gy1 = int(max(xs)) + x0, int(max(ys)) + y0
                w, h = gx1 - gx0, gy1 - gy0
                if h < 8 or h > 90 or w > 90 or w < 3:
                    continue
                key = (t, gx0 // 6, gy0 // 6)
                if key in seen:
                    continue
                seen.add(key)
                cx, cy = (gx0 + gx1) // 2, (gy0 + gy1) // 2
                if inside_any[min(H - 1, max(0, cy)), min(W - 1, max(0, cx))]:
                    continue                          # inside an editable box: never a number label
                cands.append({"digit": t, "x": gx0, "y": gy0, "w": w, "h": h, "conf": float(conf), "pass": name})

    def evaluate() -> None:
        best.clear()
        rejected.clear()
        for c in cands:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            reach = int(1.3 * h)
            ex0, ex1 = max(0, x - reach), min(W, x + w + reach)
            ey0, ey1 = max(0, y - int(0.25 * h)), min(H, y + h + int(0.25 * h))
            region = ink_all[ey0:ey1, ex0:ex1].copy()
            region[max(0, y - 4 - ey0):y + h + 4 - ey0, max(0, x - 4 - ex0):x + w + 4 - ex0] = False
            neighbours = int(region.sum())
            c["neighbour_ink"] = neighbours
            if neighbours > max(12, 0.10 * w * h):
                rejected.append({"digit": c["digit"], "x": x, "y": y, "why": f"other ink next to it ({neighbours}px)"})
                continue
            if c["digit"] not in best or c["conf"] > best[c["digit"]]["conf"]:
                best[c["digit"]] = c

    for ti in range(6):
        add_candidates(ti)
        evaluate()
        passes_used.append(str(ti + 1))
        if all(d in best for d in NUMBER_LABEL_DIGITS):
            break
    msgs.append(f"number search used {len(passes_used)} OCR pass(es)")
    for r in rejected:
        if r["digit"] not in best:
            msgs.append(f"number {r['digit']}: candidate at x={r['x']} y={r['y']} rejected - {r['why']}")

    # Fallback: OCR cannot read every digit.  The numbers are all the same size, stand alone and sit next to a
    # frame, so look for blobs of that size that satisfy the same stand-alone test, up to the number missing.
    missing = [d for d in NUMBER_LABEL_DIGITS if d not in best]
    if missing:
        hs = sorted(c["h"] for c in best.values())
        ref_h = hs[len(hs) // 2] if hs else 60
        ws = sorted(c["w"] for c in best.values())
        ref_w = ws[len(ws) // 2] if ws else 40
        blob = cv2.morphologyEx(ink_all.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        blob[inside_any] = 0
        bn, blab, bst, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
        taken = [(c["x"], c["y"], c["w"], c["h"]) for c in best.values()]
        pool: list[tuple] = []
        stage = {"size": 0, "solid": 0, "near frame": 0, "not already found": 0, "stand-alone": 0}
        for k in range(1, bn):
            x, y, w, h, area = (int(v) for v in bst[k])
            if not (NUMBER_MIN_HEIGHT <= h <= NUMBER_MAX_HEIGHT) or not (0.3 * h <= w <= 1.1 * h):
                continue
            stage["size"] += 1
            if area < 0.12 * w * h:
                continue
            stage["solid"] += 1
            gap = min((tm._rect_gap((x, y, w, h), (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))) for b in boxes),
                      default=1e9)
            if gap > margin:
                continue
            stage["near frame"] += 1
            if any(abs(x - tx) < ref_w and abs(y - ty) < ref_h for tx, ty, tw, th in taken):
                continue
            stage["not already found"] += 1
            reach = int(1.3 * h)
            ex0, ex1 = max(0, x - reach), min(W, x + w + reach)
            ey0, ey1 = max(0, y - int(0.25 * h)), min(H, y + h + int(0.25 * h))
            region = ink_all[ey0:ey1, ex0:ex1].copy()
            region[max(0, y - 4 - ey0):y + h + 4 - ey0, max(0, x - 4 - ex0):x + w + 4 - ex0] = False
            nb = int(region.sum())
            if nb > max(16, 0.25 * w * h):
                continue
            stage["stand-alone"] += 1
            pool.append((nb, gap, x, y, w, h))
        pool.sort()
        msgs.append("size-match search: " + ", ".join(f"{k}={v}" for k, v in stage.items()))
        for d, (nb, gap, x, y, w, h) in zip(missing, pool):
            best[d] = {"digit": d, "x": x, "y": y, "w": w, "h": h, "conf": 0.0, "pass": "shape match (digit not read)"}
        if pool:
            msgs.append(f"{min(len(pool), len(missing))} label(s) located by size/position because OCR could not read them")

    for d in NUMBER_LABEL_DIGITS:
        c = best.get(d)
        if c is None:
            msgs.append(f"number {d}: not found")
            continue
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        # OCR boxes are approximate (sometimes too small), so work in a wider window and take every
        # connected glyph piece that touches the OCR box - the whole digit, not just what the box covers.
        P = max(14, int(0.8 * max(w, h)))
        x0, y0, x1, y1 = max(0, x - P), max(0, y - P), min(W, x + w + P), min(H, y + h + P)
        core = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        core[max(0, y - 3 - y0):y + h + 3 - y0, max(0, x - 3 - x0):x + w + 3 - x0] = True
        usable = ~ann_near[y0:y1, x0:x1] & ~inside_any[y0:y1, x0:x1]
        ring = usable & ~core
        patch = image_rgb[y0:y1, x0:x1].astype(np.float32)
        bg = np.median(patch[ring], axis=0) if ring.sum() >= 20 else np.median(patch.reshape(-1, 3), axis=0)
        differs = (np.abs(patch - bg).max(axis=2) > 18.0) & ~inside_any[y0:y1, x0:x1]
        if int((differs & core).sum()) < 6:
            msgs.append(f"number {d}: located at x={x} y={y} but no glyph pixels differ from the background")
            continue
        merged = cv2.dilate(differs.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        gn, glab, gst, _ = cv2.connectedComponentsWithStats(merged.astype(np.uint8), connectivity=8)
        take = np.zeros_like(differs)
        limit = 2.5 * max(w, h) + 10
        hit = set(int(v) for v in np.unique(glab[core & merged]) if v != 0)
        for gi in hit:
            gx, gy, gw, gh, _a = (int(v) for v in gst[gi])
            touches_edge = gx == 0 or gy == 0 or gx + gw >= (x1 - x0) or gy + gh >= (y1 - y0)
            if touches_edge or gw > limit or gh > limit:
                continue                                   # a larger structure (frame, panel edge) - not the digit
            take |= (glab == gi) & differs
        take |= differs & core                              # always at least what the OCR box covers
        diff = cv2.dilate(take.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        diff &= usable | take
        out[y0:y1, x0:x1] |= diff
        found.append({"digit": d, "x": x, "y": y, "w": w, "h": h, "conf": round(c["conf"], 2),
                      "pixels": int(diff.sum())})
        msgs.append(f"number {d}: removed at x={x} y={y} w={w} h={h} (conf {c['conf']:.2f}, {c.get('pass', 'ocr')})")
    return out, found, msgs


def process_image() -> None:
    """Main processing function."""

    input_path = SCRIPT_DIR / INPUT_FILENAME
    output_path = SCRIPT_DIR / OUTPUT_FILENAME

    if not input_path.is_file():
        error(f"Input image not found: {input_path}")
        return

    # Load image (BGR for the detection code, RGB copy is the immutable reference)
    image_cv = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_cv is None:
        error(f"Unable to load image: {input_path}")
        return

    height, width = image_cv.shape[:2]
    info(f"Loaded image: {input_path}")
    info(f"Image resolution: {width} x {height}")

    original_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    original_rgb.setflags(write=False)

    # Get font
    try:
        font_path = get_local_font()
    except FileNotFoundError as exc:
        error(str(exc))
        return

    # Detect boxes (full-image coordinates; each box also has an "inner" interior)
    boxes = find_red_boxes(image_cv)
    if len(boxes) != EXPECTED_BOX_COUNT:
        error("")
        error("PROCESSING ABORTED.")
        error("No changes were made to the source image.")
        return

    if easyocr is None:
        error("easyocr is not installed (pip install easyocr).")
        return

    # Remove ONLY the red annotation pixels (frames, arrows, halos) - isolated red
    # image content and everything inside the frame interiors is left alone.
    info("Removing red annotations (frames, arrows) ...")
    red_mask = create_red_mask(image_cv)
    ann_mask, ann_info = tm.build_annotation_mask(red_mask, boxes, original_rgb, MASK_CONFIG)
    for blob in ann_info.get("preserved_isolated_red", []):
        warning(f"Red content at x={blob['x']} y={blob['y']} w={blob['w']} h={blob['h']} is not near any "
                "frame and was PRESERVED (treated as image content).")
    if ann_info.get("badges_cleaned"):
        info(f"Cleaned {len(ann_info['badges_cleaned'])} small red badge(s)/number(s).")
    for lb in ann_info.get("light_labels_removed", []):
        info(f"Removed white label at x={lb['x']} y={lb['y']} w={lb['w']} h={lb['h']} (number next to a frame).")
    cleaned_rgb = tm.remove_annotations(original_rgb, ann_mask, MASK_CONFIG)
    cleaned_bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)

    # Initialize EasyOCR
    info("Initializing EasyOCR...")
    try:
        reader = easyocr.Reader(["en"], gpu=False)
    except Exception as exc:
        error(f"EasyOCR initialization failed: {exc}")
        return

    if FIND_NUMBER_LABELS:
        info("Locating the numbered labels (digits 1-6) ...")
        num_mask, num_found, num_msgs = find_number_labels(reader, original_rgb, boxes, ann_mask)
        for m in num_msgs:
            (info if "removed" in m else warning)(m)
        try:   # picture of where each number was found (stays on your machine)
            dbg = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR).copy()
            for bx in boxes:
                cv2.rectangle(dbg, (int(bx["x"]), int(bx["y"])), (int(bx["x"] + bx["w"]), int(bx["y"] + bx["h"])),
                              (0, 255, 255), 3)
            for nf in num_found:
                cv2.rectangle(dbg, (nf["x"] - 6, nf["y"] - 6), (nf["x"] + nf["w"] + 6, nf["y"] + nf["h"] + 6),
                              (255, 0, 255), 4)
                cv2.putText(dbg, f"#{nf['digit']}", (nf["x"] - 6, max(30, nf["y"] - 14)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.6, (255, 0, 255), 4)
            scale = min(1.0, 1800.0 / dbg.shape[1])
            if scale < 1.0:
                dbg = cv2.resize(dbg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(SCRIPT_DIR / "debug_numbers.png"), dbg)
            info("Saved debug_numbers.png (yellow = frames, magenta = numbers it removed).")
        except Exception as exc:
            warning(f"could not write debug_numbers.png ({exc})")
        if num_mask.any():
            ann_mask = ann_mask | num_mask
            cleaned_rgb = tm.remove_annotations(original_rgb, ann_mask, MASK_CONFIG)
            cleaned_bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)

    # Measure every box (tight text mask) and OCR it, from the cleaned image
    analyses: list[tm.InteriorAnalysis] = []
    original_texts: list[str] = []
    for index, box in enumerate(boxes, start=1):
        ix, iy, iw, ih = tm.inner_rect_of(box)
        analyses.append(tm.analyze_interior(cleaned_rgb[iy:iy + ih, ix:ix + iw], MASK_CONFIG,
                                            TARGET_SPANS.get(index)))
        text = ORIGINAL_TEXT_OVERRIDES.get(index) or read_box_text(reader, cleaned_bgr, box)
        if not text and analyses[-1].found and PROMPT_FOR_MISSING_ORIGINAL:
            try:
                text = " ".join(input(
                    f"Box #{index}: OCR found no text. Type the ORIGINAL text exactly "
                    "(or ENTER to skip): ").split())
            except (EOFError, KeyboardInterrupt):
                print()
                text = ""
        original_texts.append(text)

    # Font calibration (font identity first, spacing/baseline afterwards)
    font_dirs = list(FONT_SEARCH_DIRS) + list(SYSTEM_FONT_DIRS)
    configs = _calibrate(cleaned_rgb, boxes, original_texts, font_path, analyses, font_dirs)
    configs = _maybe_download_and_recalibrate(configs, cleaned_rgb, boxes, original_texts, font_path,
                                              analyses, font_dirs)

    # Interactive replacement
    print()
    print("=" * 90)
    print("INTERACTIVE TEXT REPLACEMENT")
    print("=" * 90)
    print(f"Detected exactly {len(boxes)} target boxes.")
    print("Enter replacement text for each box.")
    print("Press ENTER to leave a box unchanged.")
    print("Replacement text is forced to a single line.")
    print("=" * 90)
    print()

    working_rgb = cleaned_rgb.copy()
    edits: list[tm.EditResult] = []
    per_box: list[dict] = []

    for index, box in enumerate(boxes, start=1):
        rect = tm.inner_rect_of(box)
        ix, iy, iw, ih = rect
        an = analyses[index - 1]
        typo = configs[index - 1]
        original_text = original_texts[index - 1]

        print(f"\nBOX #{index}")
        print(f"Location: x={box['x']} y={box['y']} w={box['w']} h={box['h']}  "
              f"(interior x={ix} y={iy} w={iw} h={ih})")
        print(f"OCR detected: {original_text}" if original_text else "OCR detected: [no text]")
        print(f"Font: {Path(typo.get('font_path') or '').name or '(none)'}  "
              f"size={typo.get('font_size')}px  confidence={typo.get('calibration_confidence')}")

        record: dict = {
            "box_index": index, "outer_rect": [box["x"], box["y"], box["w"], box["h"]],
            "inner_rect": list(rect), "status": "unchanged", "reason": "",
            "font": Path(typo.get("font_path") or "").name, "font_size_px": typo.get("font_size"),
            "calibration_confidence": typo.get("calibration_confidence"),
            "original_ink_bbox_abs": _bbox_abs(an.ink_bbox_rel, rect),
            "original_text": original_text,
            "analysis_components": an.components,      # geometry/status only, no content
            "analysis_notes": an.notes,
            "removal_pixels": int(an.removal_mask.sum()),
            "interior_background_rgb": list(an.bg_rgb), "interior_text_rgb": list(an.fg_rgb),
            "solid_background": bool(an.solid_background),
            "interior_palette": tm.color_palette(cleaned_rgb[iy:iy + ih, ix:ix + iw]),
        }

        try:
            replacement = input("Replacement text: ")
        except (EOFError, KeyboardInterrupt):
            print()
            warning("Input cancelled. Nothing was saved.")
            return

        replacement = " ".join(replacement.strip().splitlines()).strip()
        if replacement == "":
            info(f"Box #{index} left unchanged.")
            per_box.append(record)
            continue

        print(f"Replacing box #{index} with: {replacement}")
        record["replacement_text"] = replacement

        if (CENTER_SHORT_TEXT_IN_SQUARE_BOX and len(replacement) <= SHORT_TEXT_MAX_CHARS
                and 0.8 <= iw / max(ih, 1) <= 1.25):
            typo = dict(typo)
            typo["anchor"] = "interior_center"
            info(f"Box #{index}: short text in a square box - centring it in the box interior.")
        try:
            try:
                plan = tm.plan_replacement(replacement, an, (iw, ih), typo, MASK_CONFIG)
            except tm.PlacementError:
                if ON_OVERFLOW == "shrink" and typo.get("font_path"):
                    plan = _shrink_plan(replacement, an, (iw, ih), typo)
                else:
                    raise
        except (tm.PlacementError, FileNotFoundError, OSError) as exc:
            error(f"Box #{index} REJECTED - {exc}")
            error(f"Box #{index} was left unchanged (annotations removed only).")
            record.update({"status": "rejected", "reason": str(exc)})
            per_box.append(record)
            continue

        res = tm.execute_edit(working_rgb, rect, an, plan, MASK_CONFIG)
        edits.append(res)
        if res.residual_ink_pixels > 25:
            warning(f"Box #{index}: {res.residual_ink_pixels} px of old-text-like ink may remain after removal "
                    "(check this box in the output).")
        info(f"Box #{index}: {plan.font_used} {plan.size_px}px, tracking {plan.tracking_em:+.3f}em, "
             f"{plan.alignment}-aligned, baseline y={plan.baseline_rel + iy} ({'; '.join(plan.notes)})")
        record.update({
            "status": "replaced", "font": plan.font_used, "font_size_px": plan.size_px,
            "tracking_em": plan.tracking_em, "alignment": plan.alignment,
            "baseline_abs_y": plan.baseline_rel + iy,
            "replacement_ink_bbox_abs": _bbox_abs(plan.ink_bbox_rel, rect),
            "plan_notes": plan.notes,
            "residual_old_ink_pixels": res.residual_ink_pixels,
            "removal_mask": res.removal_mask, "replacement_mask": res.replacement_mask,
            "allowed_mask": res.allowed_mask,
        })
        per_box.append(record)

    # ---- Pixel-preservation proof: refuse to save if anything unauthorised changed ----
    allowed = tm.build_allowed_mask(original_rgb.shape[:2], ann_mask, edits)
    allowed_by_box = []
    for e in edits:
        full = np.zeros(original_rgb.shape[:2], dtype=bool)
        x, y, w, h = e.inner_rect
        full[y:y + h, x:x + w] = e.allowed_mask
        owner = next(b for b in boxes if tm.inner_rect_of(b) == e.inner_rect)
        allowed_by_box.append((owner, full))
    problems = tm.verify_within_boxes(allowed_by_box, original_rgb.shape[:2])
    validation = tm.validate_preservation(original_rgb, working_rgb, allowed, raise_on_fail=False)
    validation["box_boundary_problems"] = problems

    if problems or not validation["passed"]:
        error("")
        error("PRESERVATION CHECK FAILED - output NOT saved.")
        for p in problems:
            error(p)
        if not validation["passed"]:
            error(f"{validation['unauthorized_changed_pixels']} pixel(s) changed outside the authorised mask; "
                  f"first at {validation.get('first_unauthorized_pixels_xy', [])[:3]}")
        _write_reports(original_rgb, working_rgb, allowed, boxes, per_box, configs, validation, width, height)
        return

    # Save clean output (lossless PNG)
    try:
        Image.fromarray(working_rgb).save(str(output_path), format="PNG")
    except Exception as exc:
        error(f"Failed to save output image: {exc}")
        return

    _write_reports(original_rgb, working_rgb, allowed, boxes, per_box, configs, validation, width, height)

    print()
    print("=" * 90)
    print("PROCESSING COMPLETE")
    print("=" * 90)
    info(f"Output saved to: {output_path}")
    info(f"Pixel preservation: PASSED ({validation['changed_pixels']} pixels changed, all inside the "
         f"authorised mask; {validation['image_pixels'] - validation['allowed_pixels']} pixels outside it identical)")
    for rec in per_box:
        info(f"Box #{rec['box_index']}: {rec['status']}" + (f" - {rec['reason']}" if rec["reason"] else ""))
    print("=" * 90)


def _write_reports(original_rgb, working_rgb, allowed, boxes, per_box, configs, validation, width, height) -> None:
    """Redacted summary (safe to share) first, then the full diagnostics (contain client content)."""
    import traceback
    out_dir = SCRIPT_DIR / DIAGNOSTICS_DIRNAME
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warning(f"Could not create {out_dir}: {exc}")
        return

    redacted = {
        "image_size": [width, height],
        "validation": validation,
        "boxes": [{k: r.get(k) for k in (
            "box_index", "outer_rect", "inner_rect", "status", "reason", "font", "font_size_px",
            "tracking_em", "alignment", "calibration_confidence", "original_ink_bbox_abs",
            "replacement_ink_bbox_abs", "baseline_abs_y", "plan_notes", "analysis_components",
            "analysis_notes", "removal_pixels", "interior_background_rgb", "interior_text_rgb",
            "solid_background", "residual_old_ink_pixels", "interior_palette")} for r in per_box],
        "calibration": [{k: c.get(k) for k in (
            "box_index", "font_family", "font_path", "font_size", "stroke_width", "tracking_em",
            "calibration_confidence", "font_match_score", "original_ink_width_px",
            "original_ink_height_px", "bold_target")} for c in configs],
    }
    try:
        (out_dir / "redacted_summary.json").write_text(
            json.dumps(redacted, indent=2, default=tm._json_default), encoding="utf-8")
        info(f"Redacted summary written to: {out_dir / 'redacted_summary.json'}")
    except Exception as exc:
        warning(f"Could not write redacted_summary.json: {type(exc).__name__}: {exc}")

    if not WRITE_DIAGNOSTICS:
        return
    report = {"image_size": [width, height], "validation": validation,
              "boxes": [{k: v for k, v in r.items() if not k.endswith("_mask")} for r in per_box],
              "calibration": configs}
    try:
        tm.write_diagnostics(out_dir, original_rgb, working_rgb, allowed, boxes, per_box, report)
        info(f"Diagnostics written to: {out_dir}")
        warning("diagnostics/ contains crops of your source image and replacement_report.json contains the "
                "OCR/replacement text. Only redacted_summary.json is free of client content.")
    except Exception as exc:
        warning(f"Could not write full diagnostics: {type(exc).__name__}: {exc}")
        traceback.print_exc()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        process_image()
    except KeyboardInterrupt:
        print()
        warning("Processing cancelled by user.")
    except Exception as exc:
        print()
        error("UNEXPECTED ERROR")
        error(f"{type(exc).__name__}: {exc}")
        error("The source image was not intentionally modified.")
