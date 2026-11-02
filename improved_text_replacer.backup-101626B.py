#!/usr/bin/env python3
"""
Interactive Image Text Replacer
================================

Purpose:
    Detect six hollow red rectangles, remove red annotations against a
    solid-color background, and interactively replace text inside each
    target rectangle.

Features:
    - Source image is never overwritten.
    - Red HSV mask with anti-aliasing tolerance.
    - Nested-contour rectangle detection.
    - Duplicate suppression and candidate diagnostics.
    - Exactly six target regions required before editing.
    - Debug detection image, red mask, and candidate report.
    - EasyOCR recognition of existing text.
    - Local font discovery beside this script or in ./fonts.
    - Single-line replacement with dynamic font fitting.
    - Original text color estimation where possible.
    - Solid-background annotation removal.
    - Separate PNG output.

Requirements:
    pip install opencv-python numpy easyocr pillow

Important:
    This version assumes the annotations are red in the actual input image.
    The supplied sample layouts illustrate the locations and shapes.
"""

from __future__ import annotations

import glob
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import easyocr
import numpy as np
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

# Set a filename to force use of a particular local font.
# Example: FONT_FILENAME = "Roboto-Medium.ttf"
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

# Red hue ranges in OpenCV HSV format.
RED_LOWER_1 = np.array([0, 55, 45], dtype=np.uint8)
RED_UPPER_1 = np.array([13, 255, 255], dtype=np.uint8)

RED_LOWER_2 = np.array([163, 55, 45], dtype=np.uint8)
RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

# Morphology used for the detection mask.
MORPH_KERNEL_SIZE = 3
MORPH_CLOSE_ITERATIONS = 1

# Expand the red mask when removing annotations, to catch anti-aliased edges.
CLEANUP_DILATION_SIZE = 3
CLEANUP_DILATION_ITERATIONS = 1

# Rectangle geometry limits.
MIN_BOX_WIDTH = 30
MIN_BOX_HEIGHT = 25
MAX_BOX_IMAGE_AREA_RATIO = 0.15

MIN_RECTANGULARITY = 0.68
MIN_SOLIDITY = 0.80

MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 12.0

# Four-sided rectangles are expected, but anti-aliasing may change contour
# approximation slightly.
MIN_POLYGON_VERTICES = 4
MAX_POLYGON_VERTICES = 10

# Duplicate contour suppression.
DUPLICATE_IOU_THRESHOLD = 0.78

# Text fitting.
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 500
TEXT_MARGIN_RATIO = 0.04

# Solid-background color sampling.
BACKGROUND_BORDER_THICKNESS = 12

# OCR configuration.
OCR_LANGUAGES = ["en"]
OCR_GPU = False


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
    """Find a font beside the script or in its fonts/ directory."""

    if FONT_FILENAME:
        for directory in FONT_SEARCH_DIRS:
            candidate = directory / FONT_FILENAME

            if candidate.is_file():
                info(f"Using configured local font: {candidate}")
                return str(candidate)

        raise FileNotFoundError(
            f"Configured font {FONT_FILENAME!r} was not found in:\n"
            + "\n".join(str(directory) for directory in FONT_SEARCH_DIRS)
        )

    for preferred in PREFERRED_FONT_NAMES:
        for directory in FONT_SEARCH_DIRS:
            candidate = directory / preferred

            if candidate.is_file():
                info(f"Automatically selected local font: {candidate}")
                return str(candidate)

    font_files: list[str] = []

    for directory in FONT_SEARCH_DIRS:
        if directory.is_dir():
            font_files.extend(glob.glob(str(directory / "*.ttf")))
            font_files.extend(glob.glob(str(directory / "*.otf")))

    font_files = list(dict.fromkeys(font_files))

    if not font_files:
        raise FileNotFoundError(
            "No .ttf or .otf fonts found beside the script or in fonts/."
        )

    def font_rank(path: str) -> tuple[int, int, int, str]:
        name = Path(path).stem.lower()

        return (
            0 if "medium" in name else 1,
            0 if "regular" in name else 1,
            0 if "sans" in name else 1,
            name,
        )

    selected = sorted(font_files, key=font_rank)[0]

    info(f"Automatically selected local font: {selected}")
    return selected


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=max(1, int(size)))


# =============================================================================
# RED MASK AND BACKGROUND
# =============================================================================

def create_red_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Detect red annotations using two HSV hue ranges."""

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MORPH_CLOSE_ITERATIONS,
    )

    return mask


def estimate_solid_background_bgr(
    image_bgr: np.ndarray,
) -> tuple[int, int, int]:
    """
    Estimate the background color from the image border.

    A median is used instead of sampling a single pixel, making the
    estimate less sensitive to compression noise and isolated marks.
    """

    height, width = image_bgr.shape[:2]

    thickness = max(
        1,
        min(
            BACKGROUND_BORDER_THICKNESS,
            height // 4,
            width // 4,
        ),
    )

    border_parts = [
        image_bgr[:thickness, :, :],
        image_bgr[-thickness:, :, :],
        image_bgr[:, :thickness, :],
        image_bgr[:, -thickness:, :],
    ]

    pixels = np.concatenate(
        [part.reshape(-1, 3) for part in border_parts],
        axis=0,
    )

    median = np.median(pixels, axis=0).astype(np.uint8)

    return tuple(int(value) for value in median)


def create_annotation_cleanup_mask(red_mask: np.ndarray) -> np.ndarray:
    """
    Expand the red mask slightly so anti-aliased edges are removed too.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CLEANUP_DILATION_SIZE, CLEANUP_DILATION_SIZE),
    )

    return cv2.dilate(
        red_mask,
        kernel,
        iterations=CLEANUP_DILATION_ITERATIONS,
    )


def remove_red_annotations(
    image_bgr: np.ndarray,
    cleanup_mask: np.ndarray,
    background_bgr: tuple[int, int, int],
) -> np.ndarray:
    """
    Replace red annotation pixels with the estimated solid background.

    This removes detected red boxes, arrows, and red numbering without
    painting over the entire image.
    """

    result = image_bgr.copy()
    result[cleanup_mask > 0] = background_bgr

    return result


# =============================================================================
# CONTOUR METRICS
# =============================================================================

def calculate_candidate_metrics(
    contour: np.ndarray,
    image_width: int,
    image_height: int,
    contour_index: int,
    hierarchy: np.ndarray,
) -> dict:
    """Calculate geometric features for a contour candidate."""

    x, y, w, h = cv2.boundingRect(contour)

    contour_area = float(cv2.contourArea(contour))
    bounding_area = float(max(w * h, 1))

    rectangularity = contour_area / bounding_area

    hull = cv2.convexHull(contour)
    hull_area = float(max(cv2.contourArea(hull), 1))
    solidity = contour_area / hull_area

    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:
        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )
        vertices = len(approximation)
    else:
        vertices = 0

    aspect = w / max(h, 1)

    image_area = max(image_width * image_height, 1)
    image_area_ratio = bounding_area / image_area

    parent_index = int(hierarchy[contour_index][3])
    has_parent = parent_index >= 0

    return {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": contour_area,
        "bounding_area": bounding_area,
        "rect": rectangularity,
        "solid": solidity,
        "fill": rectangularity,
        "aspect": aspect,
        "vertices": vertices,
        "image_area_ratio": image_area_ratio,
        "contour_index": contour_index,
        "has_parent": has_parent,
    }


def candidate_is_valid(metrics: dict) -> bool:
    """Reject contours that do not resemble a target rectangle."""

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

    if not (
        MIN_ASPECT_RATIO
        <= metrics["aspect"]
        <= MAX_ASPECT_RATIO
    ):
        return False

    if not (
        MIN_POLYGON_VERTICES
        <= metrics["vertices"]
        <= MAX_POLYGON_VERTICES
    ):
        return False

    return True


def score_candidate(metrics: dict) -> float:
    """Rank rectangular contours, favoring clean rectangle geometry."""

    score = 0.0

    score += min(metrics["rect"], 1.0) * 40.0
    score += min(metrics["solid"], 1.0) * 25.0

    if metrics["vertices"] == 4:
        score += 20.0
    elif metrics["vertices"] in (5, 6):
        score += 12.0
    else:
        score += 5.0

    # An interior contour is useful because an arrow can distort the
    # outer contour of a rectangle to which the arrow is connected.
    if metrics["has_parent"]:
        score += 8.0

    return score


# =============================================================================
# DUPLICATE SUPPRESSION
# =============================================================================

def intersection_over_union(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]

    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def suppress_duplicate_candidates(candidates: list[dict]) -> list[dict]:
    """
    Remove nested contours describing the same rectangle.

    Smaller, well-defined interior contours are preferred over their
    larger outer-border counterparts.
    """

    ordered = sorted(
        candidates,
        key=lambda item: (
            -item["score"],
            item["w"] * item["h"],
        ),
    )

    selected: list[dict] = []

    for candidate in ordered:
        duplicate = any(
            intersection_over_union(candidate, existing)
            >= DUPLICATE_IOU_THRESHOLD
            for existing in selected
        )

        if not duplicate:
            selected.append(candidate)

    return selected


# =============================================================================
# DEBUG OUTPUT
# =============================================================================

def create_debug_image(
    image_bgr: np.ndarray,
    all_candidates: list[dict],
    selected_boxes: list[dict],
) -> None:
    """
    Save an annotated diagnostic image.

    Red: valid rectangle candidates.
    Green: selected final target regions.
    """

    debug = image_bgr.copy()

    for index, candidate in enumerate(all_candidates, start=1):
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        cv2.rectangle(
            debug,
            (x, y),
            (x + w - 1, y + h - 1),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            debug,
            f"C{index} S={candidate['score']:.0f}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for index, candidate in enumerate(selected_boxes, start=1):
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        cv2.rectangle(
            debug,
            (x, y),
            (x + w - 1, y + h - 1),
            (0, 255, 0),
            6,
        )

        cv2.putText(
            debug,
            f"TARGET {index}",
            (x, min(image_bgr.shape[0] - 10, y + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    path = SCRIPT_DIR / DEBUG_DETECTION_FILENAME

    if not cv2.imwrite(str(path), debug):
        warning(f"Could not save detection debug image: {path}")
    else:
        info(f"Detection debug image saved to: {path}")


def print_candidate_report(candidates: list[dict]) -> None:
    print()
    print("=" * 100)
    print("RECTANGLE CANDIDATE DIAGNOSTICS")
    print("=" * 100)

    if not candidates:
        print("No rectangle candidates passed the geometry filters.")

    for index, candidate in enumerate(candidates, start=1):
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
            f"vertices={candidate['vertices']} | "
            f"nested={candidate['has_parent']} | "
            f"score={candidate['score']:.2f}"
        )

    print("=" * 100)
    print()


# =============================================================================
# RECTANGLE DETECTION
# =============================================================================

def find_red_boxes(image_bgr: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """
    Detect the six hollow rectangle interiors.

    RETR_TREE returns nested contours. The interior of a hollow rectangle
    can remain detectable even when its outer border touches an arrow.
    """

    height, width = image_bgr.shape[:2]

    red_mask = create_red_mask(image_bgr)

    mask_path = SCRIPT_DIR / DEBUG_MASK_FILENAME

    if not cv2.imwrite(str(mask_path), red_mask):
        warning(f"Could not save red mask: {mask_path}")
    else:
        info(f"Red mask saved to: {mask_path}")

    projection_path = SCRIPT_DIR / DEBUG_PROJECTION_FILENAME

    # Keep this diagnostic artifact for comparison with prior runs.
    if not cv2.imwrite(str(projection_path), red_mask):
        warning(f"Could not save projection diagnostic: {projection_path}")

    contours, hierarchy_data = cv2.findContours(
        red_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy_data is None or not contours:
        info("No contours were found in the red mask.")
        create_debug_image(image_bgr, [], [])
        return [], red_mask

    hierarchy = hierarchy_data[0]

    info(f"Found {len(contours)} total contours.")

    all_metrics: list[dict] = []
    candidates: list[dict] = []

    for index, contour in enumerate(contours):
        metrics = calculate_candidate_metrics(
            contour,
            width,
            height,
            index,
            hierarchy,
        )

        all_metrics.append(metrics)

        if candidate_is_valid(metrics):
            candidate = dict(metrics)
            candidate["score"] = score_candidate(metrics)
            candidates.append(candidate)

    print_candidate_report(candidates)

    # Suppress inner/outer contour duplicates.
    selected = suppress_duplicate_candidates(candidates)

    # Stable ordering for interactive entry.
    selected.sort(
        key=lambda item: (
            item["y"],
            item["x"],
        )
    )

    create_debug_image(
        image_bgr,
        candidates,
        selected,
    )

    info(f"Selected {len(selected)} final rectangle candidates.")

    if len(selected) != EXPECTED_BOX_COUNT:
        error("")
        error("TEXT BOX DETECTION FAILED")
        error(f"Expected: {EXPECTED_BOX_COUNT} boxes")
        error(f"Detected: {len(selected)} boxes")
        error("")
        error("The source image was NOT modified.")
        error("Inspect debug_detection.png and debug_red_mask.png.")
        return [], red_mask

    # Give each target a stable one-based ID.
    for index, box in enumerate(selected, start=1):
        box["target_id"] = index

    return selected, red_mask


# =============================================================================
# OCR
# =============================================================================

def read_box_text(
    reader: easyocr.Reader,
    image_bgr: np.ndarray,
    box: dict,
) -> str:
    """Read the existing text within the detected rectangle interior."""

    x, y, w, h = (
        box["x"],
        box["y"],
        box["w"],
        box["h"],
    )

    crop = image_bgr[y:y + h, x:x + w]

    if crop.size == 0:
        return ""

    results = reader.readtext(
        crop,
        detail=1,
        paragraph=False,
    )

    if not results:
        return ""

    results.sort(
        key=lambda result: (
            min(point[1] for point in result[0]),
            min(point[0] for point in result[0]),
        )
    )

    parts = [
        str(result[1]).strip()
        for result in results
        if str(result[1]).strip()
    ]

    return " ".join(parts)


# =============================================================================
# TEXT COLOR AND FITTING
# =============================================================================

def choose_text_color(
    original_rgb: np.ndarray,
    red_mask: np.ndarray,
    box: dict,
    background_bgr: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    Estimate the original text color from non-background pixels inside
    the box. If no usable text pixels exist, choose a contrasting color.
    """

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    crop = original_rgb[y:y + h, x:x + w]

    if crop.size:
        red_crop = red_mask[y:y + h, x:x + w] > 0

        background_rgb = np.array(
            background_bgr[::-1],
            dtype=np.int16,
        )

        difference = np.max(
            np.abs(crop.astype(np.int16) - background_rgb),
            axis=2,
        )

        usable = (~red_crop) & (difference >= 35)

        pixels = crop[usable]

        if len(pixels) > 0:
            # Quantization reduces the impact of anti-aliased edge colors.
            quantized = (pixels // 16) * 16

            color, count = Counter(
                map(tuple, quantized.tolist())
            ).most_common(1)[0]

            if count >= 1:
                return tuple(
                    int(min(255, value + 8))
                    for value in color
                )

    # Fallback: black text on a light background, white on a dark one.
    b, g, r = background_bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r

    return (0, 0, 0) if luminance >= 128 else (255, 255, 255)


def text_dimensions(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)

    return (
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    )


def fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    max_width: int,
    max_height: int,
) -> int:
    """Find the largest font size that fits on a single line."""

    if not text:
        return MIN_FONT_SIZE

    margin_x = max(1, round(max_width * TEXT_MARGIN_RATIO))
    margin_y = max(1, round(max_height * TEXT_MARGIN_RATIO))

    available_width = max(1, max_width - 2 * margin_x)
    available_height = max(1, max_height - 2 * margin_y)

    low = MIN_FONT_SIZE
    high = MAX_FONT_SIZE
    best = MIN_FONT_SIZE

    while low <= high:
        middle = (low + high) // 2

        try:
            font = load_font(font_path, middle)
        except OSError:
            high = middle - 1
            continue

        width, height = text_dimensions(draw, text, font)

        if width <= available_width and height <= available_height:
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    return best


# =============================================================================
# REPLACEMENT RENDERING
# =============================================================================

def replace_text_in_box(
    working_image: Image.Image,
    original_rgb: np.ndarray,
    original_bgr: np.ndarray,
    red_mask: np.ndarray,
    box: dict,
    replacement_text: str,
    font_path: str,
    background_bgr: tuple[int, int, int],
) -> None:
    """
    Clear the rectangle interior and render the replacement as one line.

    The surrounding annotation is removed separately using the cleanup
    mask, so this function does not need to erase the box's outer border.
    """

    replacement_text = str(replacement_text or "")
    replacement_text = " ".join(replacement_text.splitlines())
    replacement_text = " ".join(replacement_text.split())

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid target dimensions: {w} x {h}")

    background_rgb = tuple(int(v) for v in background_bgr[::-1])

    draw = ImageDraw.Draw(working_image)

    # Clear only the interior of this target rectangle.
    draw.rectangle(
        (x, y, x + w - 1, y + h - 1),
        fill=background_rgb,
    )

    if not replacement_text:
        return

    font_size = fit_font_size(
        draw,
        replacement_text,
        font_path,
        w,
        h,
    )

    font = load_font(font_path, font_size)

    text_width, text_height = text_dimensions(
        draw,
        replacement_text,
        font,
    )

    bbox = draw.textbbox(
        (0, 0),
        replacement_text,
        font=font,
    )

    # The detected region is the interior of the rectangle. Centering is
    # used because the original text alignment is not known in advance.
    text_x = x + (w - text_width) / 2 - bbox[0]
    text_y = y + (h - text_height) / 2 - bbox[1]

    text_color = choose_text_color(
        original_rgb,
        red_mask,
        box,
        background_bgr,
    )

    if text_width > w or text_height > h:
        warning(
            f"Box #{box['target_id']}: text may exceed the target region."
        )

    draw.text(
        (round(text_x), round(text_y)),
        replacement_text,
        font=font,
        fill=text_color,
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

    original_bgr = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if original_bgr is None:
        error(f"Unable to load input image: {input_path}")
        return

    height, width = original_bgr.shape[:2]

    info(f"Loaded image: {input_path}")
    info(f"Image resolution: {width} x {height}")

    try:
        font_path = get_local_font()
    except FileNotFoundError as exc:
        error(str(exc))
        return

    # Detection must succeed before any output is created.
    boxes, red_mask = find_red_boxes(original_bgr)

    if len(boxes) != EXPECTED_BOX_COUNT:
        error("PROCESSING ABORTED. No output image was written.")
        return

    background_bgr = estimate_solid_background_bgr(original_bgr)

    info(
        "Estimated solid background BGR: "
        f"{background_bgr}"
    )

    cleanup_mask = create_annotation_cleanup_mask(red_mask)

    cleanup_mask_path = SCRIPT_DIR / "debug_annotation_cleanup_mask.png"

    if not cv2.imwrite(str(cleanup_mask_path), cleanup_mask):
        warning(f"Could not save cleanup mask: {cleanup_mask_path}")

    # Preserve the original arrays for OCR and text-color analysis.
    original_rgb = cv2.cvtColor(
        original_bgr,
        cv2.COLOR_BGR2RGB,
    )

    info("Initializing EasyOCR...")

    try:
        reader = easyocr.Reader(
            OCR_LANGUAGES,
            gpu=OCR_GPU,
        )
    except Exception as exc:
        error(f"EasyOCR initialization failed: {exc}")
        return

    # Start with a copy. All red annotations are cleaned against the
    # solid background; the source image remains untouched.
    cleaned_bgr = remove_red_annotations(
        original_bgr,
        cleanup_mask,
        background_bgr,
    )

    working_image = Image.fromarray(
        cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
    )

    print()
    print("=" * 90)
    print("INTERACTIVE TEXT REPLACEMENT")
    print("=" * 90)
    print(f"Detected exactly {len(boxes)} target rectangles.")
    print("Enter replacement text for each box.")
    print("Press ENTER to keep the existing interior text.")
    print("Red annotations will be removed from the output.")
    print("Replacement text is forced to a single line.")
    print("=" * 90)

    try:
        for box in boxes:
            index = box["target_id"]

            original_text = read_box_text(
                reader,
                original_bgr,
                box,
            )

            print()
            print(f"BOX #{index}")
            print(
                f"Interior: x={box['x']} y={box['y']} "
                f"w={box['w']} h={box['h']}"
            )
            print(
                f"Geometry: aspect={box['aspect']:.3f}, "
                f"rectangularity={box['rect']:.3f}"
            )

            print(
                f"OCR detected: {original_text}"
                if original_text
                else "OCR detected: [no text]"
            )

            replacement = input("Replacement text: ")

            if replacement.strip() == "":
                info(
                    f"Box #{index}: existing interior text preserved."
                )
                continue

            replacement = " ".join(replacement.splitlines())
            replacement = " ".join(replacement.split())

            print(f"Replacement: {replacement}")

            replace_text_in_box(
                working_image=working_image,
                original_rgb=original_rgb,
                original_bgr=original_bgr,
                red_mask=red_mask,
                box=box,
                replacement_text=replacement,
                font_path=font_path,
                background_bgr=background_bgr,
            )

    except (EOFError, KeyboardInterrupt):
        print()
        warning("Interactive input cancelled. No output was saved.")
        return

    # Write to a separate file. Never overwrite original.png.
    try:
        working_image.save(
            str(output_path),
            format="PNG",
        )
    except Exception as exc:
        error(f"Failed to save output image: {exc}")
        return

    # Verify the output can be reopened and has the expected dimensions.
    verification = cv2.imread(
        str(output_path),
        cv2.IMREAD_COLOR,
    )

    if verification is None:
        error("Output verification failed: saved image could not be reopened.")
        return

    if verification.shape[:2] != original_bgr.shape[:2]:
        error("Output verification failed: image dimensions changed.")
        return

    print()
    print("=" * 90)
    print("PROCESSING COMPLETE")
    print("=" * 90)
    info(f"Output saved to: {output_path}")
    info(f"Output dimensions verified: {width} x {height}")
    info("Source image was not overwritten.")
    info(f"Detection debug image: {SCRIPT_DIR / DEBUG_DETECTION_FILENAME}")
    info(f"Red mask: {SCRIPT_DIR / DEBUG_MASK_FILENAME}")
    info(f"Cleanup mask: {cleanup_mask_path}")
    print("=" * 90)


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
        sys.exit(1)