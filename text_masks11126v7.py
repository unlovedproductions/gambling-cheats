#!/usr/bin/env python3
"""Mask-based text replacement engine used by text_replacer_complete.py.

Responsibilities (all pure numpy/OpenCV/Pillow, no OCR dependency so it can be
unit-tested):

* frame detection        - find the enclosed interior of every red frame, so
                           arrows touching a frame or stacked frames are not
                           merged into one region
* annotation mask        - which red pixels are annotation overlay (frames,
                           arrows) and which are unrelated image content
* interior analysis      - background/foreground colour, antialiased ink
                           coverage, the TIGHT text-removal mask, ink bounds
* typography rendering   - per-glyph layout (tracking, kerning), antialiased
                           alpha layer, overflow detection
* edit execution         - reconstruct only removed pixels, composite the new
                           text, return the authorised mask
* pixel preservation     - proof that nothing outside the authorised mask
                           changed

Coordinate convention
---------------------
Everything is in FULL-IMAGE pixel coordinates unless a name ends in ``_rel``
(relative to the interior rectangle's top-left).  Images handled here are RGB
uint8 arrays; text_replacer_complete.py converts from/to OpenCV's BGR once.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # fontTools is only used for glyph-coverage checks
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover
    TTFont = None


class PlacementError(Exception):
    """Replacement cannot be placed without violating a spatial constraint."""


class PreservationError(Exception):
    """Pixels outside the authorised modification mask changed."""


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MaskConfig:
    # --- interior / ink analysis ---
    ink_alpha_low: float = 0.08        # coverage above which a pixel counts as text
    noise_floor: float = 10.0          # min max-channel difference from bg to count
    min_contrast: float = 25.0         # |fg - bg| below this -> "no text found"
    edge_contact_ratio: float = 0.40   # ink touching >=40% of an interior side is NOT text
    solid_bg_std: float = 6.0          # background std below this -> plain fill, else smooth fill
    edge_ignore_px: int = 3            # outer interior band ignored for text (frame anti-alias residue lives here)
    fringe_alpha: float = 0.02         # faint anti-alias pixels touching removed glyphs are removed too
    inpaint_radius: int = 2
    # --- replacement ---
    overflow_alpha_tolerance: int = 8  # alpha (0-255) allowed outside interior before reject
    max_tracking_tighten_em: float = 0.03
    allow_clamp: bool = False          # slide an overflowing (but narrow enough) text back inside the interior
    alt_font_score_margin: float = 0.03
    # --- annotation / frames ---
    interior_guard_px: int = 2         # interior border band that annotation removal may not touch
    attach_distance_ratio: float = 0.04  # red blob within this fraction of image width of a frame = arrow
    preserve_isolated_red: bool = False  # False = remove ALL red annotation (numbers too); True keeps far-away red
    tint_threshold: int = 10           # R - (G+B)/2 above this = red halo pixel
    halo_radius_px: int = 12           # how far faint red fringe is followed outwards from the annotation
    scrub_radius_px: int = 14                # final scrub: clean everything non-bright this close to a removed annotation
    scrub_keep_delta: float = 22.0           # brighter than the local background by this much = real content, protected
    scrub_protect_px: int = 2                # soft edge kept around protected bright content
    scrub_sigma: float = 12.0
    scrub_interior_margin_px: int = 12       # the middle of each editable frame is left to the text-replacement step
    remove_frame_residue: bool = True        # remove the old frame's faint inner edge together with the old text
    ann_fringe_px: int = 5                   # also remove this much of the faint fringe around the annotation (never bright lettering)
    fringe_bright_max: float = 45.0
    mirror_width_ratio: float = 1.0          # full width of a round letter relative to its height
    faint_glow_max: float = 85.0             # glow pixels are at most this much brighter than the local background
    badge_pad_px: int = 8                    # small red badges/numbers: everything differing from the background within this margin goes
    badge_max_ratio: float = 0.045           # a red annotation component is a 'badge' if its bbox is at most this fraction of the image width
    remove_light_labels: bool = True         # remove small white label blobs (numbers) lying next to a frame, outside every interior
    light_label_delta: float = 45.0
    light_label_search_ratio: float = 0.06   # how far from a frame (fraction of image width) a white number may sit
    light_label_max_size: tuple = (60, 60)   # max (width, height) of a single number glyph
    light_label_neighbour_ratio: float = 1.6 # a blob with other light blobs within this many glyph-heights sideways is TEXT, not a number
    light_halo_delta: float = 14.0           # lighter than local background by this much = light outline pixel
    light_halo_chroma: float = 60.0          # ... and at most this colourful (white/grey, not UI colour)
    light_halo_radius_px: int = 4            # how far the light outline is followed from the annotation
    light_halo_interior_band_px: int = 5     # inside a frame, only this band along the edge may be cleaned
    fill_band_px: int = 8             # clean pixels within this distance of an annotation are used to fill it
    thin_annotation_px: float = 6.0    # strokes thinner than this (arrow shafts) use inpainting, thicker use smooth fill
    inpaint_annotation_radius: int = 3
    min_frame_inner: tuple = (20, 12)
    frame_rect_ratio: float = 0.90


# =============================================================================
# SMALL HELPERS
# =============================================================================

def inner_rect_of(box: dict) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the editable interior of a box (falls back to the box)."""
    inner = box.get("inner")
    if inner:
        return int(inner["x"]), int(inner["y"]), int(inner["w"]), int(inner["h"])
    return int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])


def _iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(ax - (bx + bw), bx - (ax + aw), 0)
    dy = max(ay - (by + bh), by - (ay + ah), 0)
    return math.hypot(dx, dy)


# =============================================================================
# FRAME DETECTION  (enclosed interiors of red rectangles)
# =============================================================================

def _stroke_thickness(fg: np.ndarray, x: int, y: int, w: int, h: int, limit: int = 400) -> int:
    """Frame stroke thickness, measured outward from the interior edges.

    Takes the MINIMUM over the four sides of the median of three samples, so an
    arrow shaft touching one side or a neighbouring frame sharing an edge does
    not inflate the result.
    """
    H, W = fg.shape

    def run(cx: int, cy: int, dx: int, dy: int) -> int:
        n = 0
        while 0 <= cx < W and 0 <= cy < H and fg[cy, cx] and n < limit:
            n += 1
            cx += dx
            cy += dy
        return n

    fr = (0.25, 0.5, 0.75)
    sides = [
        [run(x - 1, y + int(h * f), -1, 0) for f in fr],          # left
        [run(x + w, y + int(h * f), 1, 0) for f in fr],           # right
        [run(x + int(w * f), y - 1, 0, -1) for f in fr],          # top
        [run(x + int(w * f), y + h, 0, 1) for f in fr],           # bottom
    ]
    return max(1, int(min(float(np.median(s)) for s in sides)))


def detect_frames(red_mask: np.ndarray, cfg: Optional[MaskConfig] = None) -> list[dict]:
    """Find rectangular red frames from the regions they ENCLOSE.

    An arrow touching a frame, or two frames sharing an edge, merge into one
    red connected component - but each frame still encloses its own rectangular
    background hole.  Working from the holes keeps every box separate and gives
    the exact editable interior.
    """
    cfg = cfg or MaskConfig()
    fg = red_mask > 0
    H, W = fg.shape
    inv = (~fg).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    frames: list[dict] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue  # touches the image border: that is the page background
        if w < cfg.min_frame_inner[0] or h < cfg.min_frame_inner[1]:
            continue
        if area / float(w * h) < cfg.frame_rect_ratio:
            continue
        t = _stroke_thickness(fg, x, y, w, h)
        ox, oy = max(0, x - t), max(0, y - t)
        ox2, oy2 = min(W, x + w + t), min(H, y + h + t)
        frames.append({
            "x": ox, "y": oy, "w": ox2 - ox, "h": oy2 - oy,
            "inner": {"x": x, "y": y, "w": w, "h": h},
            "stroke_px": t,
            "source": "frame",
        })
    frames.sort(key=lambda f: (f["y"], f["x"]))
    return frames


def reconcile_boxes(
    legacy_boxes: list[dict],
    frames: list[dict],
    expected: int,
    notes: Optional[list[str]] = None,
) -> list[dict]:
    """Combine the original candidate logic with frame detection.

    * Legacy box that matches a frame (IoU >= 0.85) is kept as-is (its fields
      and identity preserved) and just gains an ``inner`` rectangle.
    * Legacy box that does not match any frame (arrow merged in, two frames
      merged ...) is replaced by the frame, because its bounds are wrong.
    * If frame detection does not find exactly ``expected`` frames but the
      legacy logic found ``expected`` boxes, the legacy boxes are used and an
      interior is estimated.
    """
    notes = notes if notes is not None else []
    if len(frames) == expected:
        final: list[dict] = []
        for fr in frames:
            match = None
            for lb in legacy_boxes:
                if _iou(lb, fr) >= 0.85:
                    match = lb
                    break
            if match is not None:
                box = dict(match)
                box["inner"] = dict(fr["inner"])
                box["stroke_px"] = fr["stroke_px"]
                final.append(box)
            else:
                final.append(dict(fr))
        replaced = sum(1 for b in final if b.get("source") == "frame")
        notes.append(
            f"frame detection found {len(frames)} frames; "
            f"{len(final) - replaced} legacy box(es) confirmed, {replaced} corrected/added"
        )
        final.sort(key=lambda b: (b["y"], b["x"]))
        return final

    if len(legacy_boxes) == expected:
        notes.append(
            f"frame detection found {len(frames)} frames (expected {expected}); "
            "falling back to legacy boxes with an estimated interior"
        )
        out = []
        for lb in legacy_boxes:
            box = dict(lb)
            fr = next((f for f in frames if _iou(lb, f) >= 0.85), None)
            if fr is not None:
                box["inner"] = dict(fr["inner"])
                box["stroke_px"] = fr["stroke_px"]
            else:
                pad = max(2, int(0.06 * min(lb["w"], lb["h"])))
                box["inner"] = {"x": lb["x"] + pad, "y": lb["y"] + pad,
                                "w": max(1, lb["w"] - 2 * pad), "h": max(1, lb["h"] - 2 * pad)}
                box["stroke_px"] = pad
                notes.append(f"interior of box at ({lb['x']},{lb['y']}) is only an estimate")
            out.append(box)
        return out

    notes.append(
        f"expected {expected} boxes; legacy logic found {len(legacy_boxes)}, "
        f"frame detection found {len(frames)}"
    )
    return []


# =============================================================================
# ANNOTATION MASK
# =============================================================================

def build_annotation_mask(
    red_mask: np.ndarray,
    boxes: list[dict],
    image_rgb: Optional[np.ndarray] = None,
    cfg: Optional[MaskConfig] = None,
) -> tuple[np.ndarray, dict]:
    """Boolean mask of red ANNOTATION pixels (frames, arrows, their halos).

    * red blobs within ``attach_distance_ratio * width`` of a frame are
      annotation (frame strokes, arrows, arrowheads);
    * isolated red blobs elsewhere are treated as image content and kept
      (``preserve_isolated_red``) and reported;
    * the interior of every frame (minus a guard band) is protected, so red
      text inside a frame is never mistaken for annotation;
    * antialiasing halo = red-tinted pixels within ``halo_radius_px`` of the
      annotation, so no pink outline is left behind.
    """
    cfg = cfg or MaskConfig()
    H, W = red_mask.shape[:2]
    fg = (red_mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    attach = max(10, int(cfg.attach_distance_ratio * W))
    outers = [(int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])) for b in boxes]

    # Interiors (holes of the frames).  A red component with pixels OUTSIDE every
    # interior is frame/arrow (it contains the frame stroke or an arrow shaft);
    # a red component entirely INSIDE an interior is red lettering and is kept.
    inside_any = np.zeros((H, W), dtype=bool)
    for b in boxes:
        ix, iy, iw, ih = inner_rect_of(b)
        inside_any[iy:iy + ih, ix:ix + iw] = True
    outside_count = np.bincount(lab[~inside_any].ravel(), minlength=n)

    keep = np.zeros(n, dtype=bool)
    preserved: list[dict] = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        gap = min((_rect_gap((x, y, w, h), o) for o in outers), default=1e9)
        near_frame = gap <= attach or not cfg.preserve_isolated_red
        if near_frame and outside_count[i] > 0:
            keep[i] = True            # frame stroke, arrow, arrowhead (even the tip inside the interior)
        elif not near_frame:
            preserved.append({"x": x, "y": y, "w": w, "h": h, "gap_to_nearest_box_px": round(gap, 1)})

    ann = keep[lab]

    # (1) Number badges: a red ring/disc with a white digit leaves the digit as an enclosed hole.
    # Fill small enclosed holes (never a frame interior) so the digit is removed with its badge.
    if ann.any():
        closed = cv2.dilate(ann.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # bridge hairline gaps in outlines
        inv = (~closed).astype(np.uint8)
        hn, hlab, hst, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        hole_max = max(2500, int(0.0015 * H * W))
        for j in range(1, hn):
            hx, hy, hw_, hh_, ha = (int(v) for v in hst[j])
            if hx == 0 or hy == 0 or hx + hw_ >= W or hy + hh_ >= H:
                continue                      # touches the image border = background, not a hole
            if ha > hole_max:
                continue
            ov = 0
            for bb in boxes:
                ix, iy, iw, ih = inner_rect_of(bb)
                ox = max(0, min(hx + hw_, ix + iw) - max(hx, ix))
                oy = max(0, min(hy + hh_, iy + ih) - max(hy, iy))
                ov = max(ov, ox * oy)
            if ov > 0.3 * hw_ * hh_:
                continue                      # a frame interior (editable area) - never fill
            ann[hlab == j] = True

    if image_rgb is not None and cfg.halo_radius_px > 0:
        r = image_rgb[..., 0].astype(np.int16)
        g = image_rgb[..., 1].astype(np.int16)
        b = image_rgb[..., 2].astype(np.int16)
        tint_raw = (r - (g + b) // 2).astype(np.float32)   # kept for reference only
        # Tint relative to the LOCAL background: on a dark green/blue UI even a faint red edge has
        # negative absolute tint, so an absolute threshold never sees it.
        far = ~cv2.dilate(ann.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool)
        wgt = far.astype(np.float32)
        sg = 25.0
        den = cv2.GaussianBlur(wgt, (0, 0), sg)
        num = cv2.GaussianBlur(tint_raw * wgt, (0, 0), sg)
        base_tint = np.where(den > 1e-3, num / np.maximum(den, 1e-3), float(np.median(tint_raw)))
        # A pixel is "reddened" only if it moved TOWARDS red relative to the local background: red clearly
        # up while green/blue did not rise with it.  White or grey pixels raise all channels together, so
        # white lettering on a dark green background is NOT mistaken for red annotation.
        def _local(ch):
            nm = cv2.GaussianBlur(ch.astype(np.float32) * wgt, (0, 0), sg)
            return np.where(den > 1e-3, nm / np.maximum(den, 1e-3), float(np.median(ch)))
        d_r = r.astype(np.float32) - _local(r)
        d_g = g.astype(np.float32) - _local(g)
        d_b = b.astype(np.float32) - _local(b)
        tint_strict = (d_r > cfg.tint_threshold) & (d_r > 2.0 * np.maximum(np.maximum(d_g, d_b), 0.0) + 4.0)
        # The soft white/pink glow around annotations: relative tint (this is what removed the faint boxes
        # before), but ONLY for faint pixels - bright lettering is far brighter than any glow and is never taken.
        lum0 = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        faint = (lum0 - _local(lum0)) < cfg.faint_glow_max
        tint = tint_strict | (((tint_raw - base_tint) > cfg.tint_threshold) & faint)
        # Follow the faint red-tinted fringe outwards (connected growth through tinted pixels only),
        # so dark-red / semi-transparent edges of frames, arrows and numbers are removed as well.
        k3 = np.ones((3, 3), np.uint8)
        grown = ann.copy()
        for _ in range(cfg.halo_radius_px):
            nxt = (cv2.dilate(grown.astype(np.uint8), k3).astype(bool) & tint) | grown
            if int(nxt.sum()) == int(grown.sum()):
                break
            grown = nxt
        ann = grown

        # (2) White / light outline around frames, arrows and numbers (annotation tools often add one).
        # Pixels that are clearly lighter than the local background and nearly neutral, connected to the
        # annotation within halo_radius_px, are removed.  Deep inside a frame interior (the editable
        # area) nothing is grown - only the band hugging the frame edge.
        lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        chroma = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)).astype(np.float32)
        far2 = ~cv2.dilate(ann.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool)
        w2 = far2.astype(np.float32)
        den2 = cv2.GaussianBlur(w2, (0, 0), 25.0)
        base_lum = np.where(den2 > 1e-3, cv2.GaussianBlur(lum * w2, (0, 0), 25.0) / np.maximum(den2, 1e-3),
                            float(np.median(lum)))
        light = ((lum - base_lum) > cfg.light_halo_delta) & (chroma <= cfg.light_halo_chroma)
        core = np.zeros((H, W), dtype=bool)
        guard = max(1, int(cfg.light_halo_interior_band_px))
        for bb in boxes:
            ix, iy, iw, ih = inner_rect_of(bb)
            if iw > 2 * guard and ih > 2 * guard:
                core[iy + guard:iy + ih - guard, ix + guard:ix + iw - guard] = True
        light &= ~core
        # Only THIN outlines (1-2 px) may be removed.  Anything at least 3 px wide is a letter stroke or UI
        # element, never an outline - so a letter touching a frame keeps its full shape.
        solid = cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        solid = cv2.dilate(solid, np.ones((3, 3), np.uint8)).astype(bool)
        light &= ~solid
        grown2 = ann.copy()
        for _ in range(int(cfg.light_halo_radius_px)):
            nxt = (cv2.dilate(grown2.astype(np.uint8), k3).astype(bool) & light) | grown2
            if int(nxt.sum()) == int(grown2.sum()):
                break
            grown2 = nxt
        ann = grown2
        # Faint outer fringe of the annotation (1-3 px of slightly different colour): remove it, but never
        # bright lettering - only pixels that are close to the local background brightness.
        if cfg.ann_fringe_px > 0:
            kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.ann_fringe_px + 1, 2 * cfg.ann_fringe_px + 1))
            ring_f = cv2.dilate(ann.astype(np.uint8), kf).astype(bool) & ~ann
            ann = ann | (ring_f & ((lum - base_lum) < cfg.fringe_bright_max))

        # (3) Small red badges / number labels: take everything that differs from the background inside
        # the badge's padded bounding box (white digits that are not fully enclosed by red are caught here).
        badge_max = max(60, int(cfg.badge_max_ratio * W))
        pad = int(cfg.badge_pad_px)
        differs = (np.abs(lum - base_lum) > 14.0) | tint
        badges: list[dict] = []
        for i in range(1, n):
            if not keep[i]:
                continue
            x, y, w, h = (int(v) for v in stats[i][:4])
            if max(w, h) > badge_max:
                continue
            x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
            win = differs[y0:y1, x0:x1] & ~core[y0:y1, x0:x1]
            ann[y0:y1, x0:x1] |= win
            badges.append({"x": x, "y": y, "w": w, "h": h})

        # (4) White numbers with no red attached: small light blobs next to a frame, outside every interior.
        labels: list[dict] = []
        if cfg.remove_light_labels:
            strong = ((lum - base_lum) > cfg.light_label_delta) & (chroma <= cfg.light_halo_chroma) & ~inside_any & ~ann
            ln, llab, lst, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), connectivity=8)
            search = max(20, int(cfg.light_label_search_ratio * W))
            mw, mh = cfg.light_label_max_size
            allb = [tuple(int(v) for v in lst[k][:4]) for k in range(1, ln) if int(lst[k][4]) >= 4]
            for k in range(1, ln):
                x, y, w, h, area = (int(v) for v in lst[k])
                if area < 12 or w > mw or h > mh or h < 8:
                    continue
                reach = cfg.light_label_neighbour_ratio * h
                crowded = False
                for (ox, oy, ow, oh) in allb:
                    if (ox, oy, ow, oh) == (x, y, w, h):
                        continue
                    sideways = max(ox - (x + w), x - (ox + ow))
                    vert = min(y + h, oy + oh) - max(y, oy)
                    if sideways <= reach and vert > 0.3 * min(h, oh):
                        crowded = True       # part of a word / text line -> real content, keep
                        break
                if crowded:
                    continue
                if min((_rect_gap((x, y, w, h), o) for o in outers), default=1e9) > search:
                    continue
                m = llab[y:y + h, x:x + w] == k
                m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                ann[y:y + h, x:x + w] |= m
                labels.append({"x": x, "y": y, "w": w, "h": h})
        return ann, {"preserved_isolated_red": preserved, "attach_distance_px": attach,
                     "badges_cleaned": badges, "light_labels_removed": labels}
    return ann, {"preserved_isolated_red": preserved, "attach_distance_px": attach}


def _smooth_fill_window(win_rgb: np.ndarray, fill: np.ndarray, valid: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised Gaussian fill of ``fill`` pixels from ``valid`` pixels. Returns (estimate, ok-mask)."""
    pix = win_rgb.astype(np.float32)
    wgt = valid.astype(np.float32)
    est = np.zeros_like(pix)
    done = np.zeros(fill.shape, dtype=bool)
    src = pix * wgt[..., None]
    sg = float(sigma)
    for _ in range(4):
        den = cv2.GaussianBlur(wgt, (0, 0), sg)
        num = np.stack([cv2.GaussianBlur(src[..., c], (0, 0), sg) for c in range(3)], axis=2)
        ok = (den > 1e-3) & ~done
        est[ok] = num[ok] / den[ok][:, None]
        done |= ok
        if done[fill].all():
            break
        sg *= 2.0
    return np.clip(np.rint(est), 0, 255).astype(np.uint8), done


def remove_annotations(image_rgb: np.ndarray, ann_mask: np.ndarray, cfg: Optional[MaskConfig] = None) -> np.ndarray:
    """Fill ONLY the annotation pixels; every other pixel is copied unchanged.

    Thin strokes (arrow shafts) are inpainted.  Thick strokes (frames, arrowheads, numbers) are
    filled from the clean pixels in a narrow band around them with a normalised Gaussian whose
    scale follows the stroke width - small-radius inpainting smears a visible outline there.
    """
    cfg = cfg or MaskConfig()
    out = image_rgb.copy()
    if not ann_mask.any():
        return out
    H, W = ann_mask.shape
    ann_u8 = ann_mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ann_u8, connectivity=8)
    band = max(2, int(cfg.fill_band_px))
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        comp_full = lab == i if n < 40 else None
        pad = 60
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        win_ann = ann_mask[y0:y1, x0:x1]
        comp = (lab[y0:y1, x0:x1] == i)
        win = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
        thick = 2.0 * float(cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3).max())
        inp = cv2.inpaint(win, win_ann.astype(np.uint8) * 255, cfg.inpaint_annotation_radius, cv2.INPAINT_TELEA)
        if thick < cfg.thin_annotation_px:
            inner_t = cv2.dilate(win_ann.astype(np.uint8), k_in).astype(bool)
            valid_t = cv2.dilate(win_ann.astype(np.uint8), k_band).astype(bool) & ~inner_t
            near_hi = np.zeros_like(comp)
            if valid_t.any():
                lum_t = win.astype(np.float32).mean(axis=2)
                hi_t = valid_t & (lum_t > float(np.median(lum_t[valid_t])) + 25.0)
                if hi_t.any():
                    near_hi = cv2.dilate(hi_t.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                    cand_t = valid_t & ~cv2.dilate(hi_t.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    if int(cand_t.sum()) > 40:
                        valid_t = cand_t
            est_t, ok_t = _smooth_fill_window(win, comp, valid_t, 3.0)
            res = np.where(ok_t[..., None], est_t, inp)
            res = np.where(near_hi[..., None], inp, res)      # beside lettering keep the inpaint (continues letter shapes)
            use = comp
        else:
            inner = cv2.dilate(win_ann.astype(np.uint8), k_in).astype(bool)
            valid = cv2.dilate(win_ann.astype(np.uint8), k_band).astype(bool) & ~inner
            sigma = float(np.clip(thick * 0.75, 3.0, 30.0))
            lum_w = win.astype(np.float32).mean(axis=2)
            valid_bg = valid
            if valid.any():
                hi = valid & (lum_w > float(np.median(lum_w[valid])) + 25.0)
                if hi.any():
                    cand = valid & ~cv2.dilate(hi.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    if int(cand.sum()) > 40:
                        valid_bg = cand          # bright lettering next to the stroke must not tint the fill
            est, ok = _smooth_fill_window(win, comp, valid_bg, sigma)
            res = np.where(ok[..., None], est, inp)
            # Where the stroke touches bright lettering, continue the letter shapes by inpainting with a
            # radius that spans the stroke (a plain blur would leave the letter cut off - an 'o' becomes a 'c').
            lumw = win.astype(np.float32).mean(axis=2)
            if valid.any():
                bright = valid & (lumw > float(np.median(lumw[valid])) + 40.0)
                if bright.any():
                    rr = 5
                    near_text = cv2.dilate(bright.astype(np.uint8), cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (13, 13))).astype(bool)
                    big = cv2.inpaint(win, win_ann.astype(np.uint8) * 255, rr, cv2.INPAINT_TELEA)
                    use_big = comp & near_text
                    res = np.where(use_big[..., None], big, res)
            use = comp
        sub = out[y0:y1, x0:x1]
        sub[use] = res[use]
    return out



def repair_clipped_letters(image_rgb: np.ndarray, ann_mask: np.ndarray, boxes: list[dict],
                           cfg: Optional[MaskConfig] = None) -> tuple[np.ndarray, list[dict]]:
    """Rebuild round letters that were cut by a removed annotation (an 'o' that now looks like a 'c').

    The cut can be a strip off one side (frame stroke) or a slit through it (arrow shaft).  A letter is repaired
    only if (a) it is lettering outside every frame interior, (b) removed-stroke pixels lie on one side of it,
    and (c) that side is OPEN like a 'c' (two arms with a gap).  Every removed-stroke pixel in the letter's
    extended box is then filled from its mirror image across the letter's vertical axis, taking the value
    only from pixels that were NOT part of the stroke.  Only stroke pixels are ever written.
    """
    cfg = cfg or MaskConfig()
    out = image_rgb.copy()
    H, W = ann_mask.shape
    done: list[dict] = []
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ink = np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 31).astype(np.int16)) > 28
    ink &= ~ann_mask
    inside = np.zeros((H, W), dtype=bool)
    for b in boxes:
        ix, iy, iw, ih = inner_rect_of(b)
        inside[iy:iy + ih, ix:ix + iw] = True
    ink &= ~inside
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)

    def runs(col: np.ndarray) -> list[tuple[int, int]]:
        res, start = [], None
        for k, v in enumerate(col):
            if v and start is None:
                start = k
            elif not v and start is not None:
                res.append((start, k)); start = None
        if start is not None:
            res.append((start, len(col)))
        return res

    for i in range(1, n):
        x, y, w, h, _a = (int(v) for v in st[i])
        if not (14 <= h <= 120) or w < 0.35 * h or w > 1.3 * h:
            continue
        comp = lab[y:y + h, x:x + w] == i
        full = max(w, int(round(cfg.mirror_width_ratio * h)))
        ys = np.arange(max(0, y - 1), min(H, y + h + 1))
        for side in ("right", "left"):
            if side == "right":
                xlo, xhi = x + w // 2, min(W, x + full + 2)
                edge = comp[:, max(0, w - 2):w].any(axis=1)
                xc = x + full / 2.0
            else:
                xlo, xhi = max(0, x + w - full - 2), x + w - w // 2
                edge = comp[:, 0:2].any(axis=1)
                xc = x + w - full / 2.0
            xs = np.arange(xlo, xhi)
            if xs.size == 0:
                continue
            if int(ann_mask[np.ix_(ys, xs)].sum()) < 10:
                continue                                     # no removed stroke on this side
            rr = runs(edge)
            if len(rr) < 2 or max(rr[k + 1][0] - rr[k][1] for k in range(len(rr) - 1)) < 0.2 * h:
                continue                                     # this side is not open like a 'c'
            written = 0
            for xx in xs:
                src = int(np.floor(2.0 * xc - xx - 1.0))
                if not (x <= src < x + w) or src == xx:
                    continue
                rows = ys[ann_mask[ys, xx] & ~ann_mask[ys, src]]   # stroke pixels whose mirror partner is real
                if rows.size:
                    out[rows, xx] = image_rgb[rows, src]
                    written += int(rows.size)
            if written:
                done.append({"x": x, "y": y, "w": w, "h": h, "side": side, "hidden_px": int(written)})
    return out, done



def scrub_annotation_zone(image_rgb: np.ndarray, ann_mask: np.ndarray, boxes: list[dict],
                          cfg: Optional[MaskConfig] = None) -> tuple[np.ndarray, np.ndarray]:
    """Final clean-up so no trace of an annotation survives, even on a bright phone screen.

    Within ``scrub_radius_px`` of every removed annotation, every pixel that is not clearly bright content is
    replaced by a smooth background estimate taken from clean pixels further out.  Bright lettering (and its
    soft edge) is protected, and the middle of each editable frame is left to the text-replacement step.
    Works from brightness against the local background, not from colour, so a leftover of ANY shade is caught.
    Returns (new image, mask of pixels that may have changed).
    """
    cfg = cfg or MaskConfig()
    H, W = ann_mask.shape
    out = image_rgb.copy()
    scrub = np.zeros((H, W), dtype=bool)
    if not ann_mask.any() or cfg.scrub_radius_px <= 0:
        return out, scrub
    r = int(cfg.scrub_radius_px)
    zone = cv2.dilate(ann_mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    core = np.zeros((H, W), dtype=bool)
    m = int(cfg.scrub_interior_margin_px)
    for b in boxes:
        ix, iy, iw, ih = inner_rect_of(b)
        if iw > 2 * m and ih > 2 * m:
            core[iy + m:iy + ih - m, ix + m:ix + iw - m] = True
    n, lab, st, _ = cv2.connectedComponentsWithStats(zone.astype(np.uint8), connectivity=8)
    sg = float(cfg.scrub_sigma)
    pad = int(3 * sg + 2 * r)
    pk = 2 * int(cfg.scrub_protect_px) + 1
    for i in range(1, n):
        x, y, w, h, _a = (int(v) for v in st[i])
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        win = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
        zc = (lab[y0:y1, x0:x1] == i)
        zall = zone[y0:y1, x0:x1]                     # any zone (a neighbour annotation must not feed the estimate)
        far = ~zall
        if int(far.sum()) < 400:
            continue
        lum = win.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        wf = far.astype(np.float32)
        den = cv2.GaussianBlur(wf, (0, 0), sg)
        base = np.where(den > 1e-3, cv2.GaussianBlur(lum * wf, (0, 0), sg) / np.maximum(den, 1e-3),
                        float(np.median(lum[far])))
        bright = (lum - base) > float(cfg.scrub_keep_delta)
        protect = cv2.dilate(bright.astype(np.uint8), np.ones((pk, pk), np.uint8)).astype(bool)
        valid = far & ~protect
        if int(valid.sum()) < 400:
            continue
        target = zc & ~protect & ~core[y0:y1, x0:x1]
        if not target.any():
            continue
        est, ok = _smooth_fill_window(win, target, valid, sg)
        use = target & ok
        sub = out[y0:y1, x0:x1]
        sub[use] = est[use]
        scrub[y0:y1, x0:x1] |= use
    return out, scrub


# =============================================================================
# INTERIOR ANALYSIS  (tight text-removal mask)
# =============================================================================

@dataclass
class InteriorAnalysis:
    found: bool
    bg_rgb: tuple[int, int, int]
    fg_rgb: tuple[int, int, int]
    contrast: float
    alpha: np.ndarray                  # float32 coverage 0..1 (h, w)
    removal_mask: np.ndarray           # bool: pixels of the TARGET text only
    excluded_mask: np.ndarray          # bool: ink kept on purpose (touches border / outside span)
    ink_bbox_rel: Optional[tuple[int, int, int, int]]   # x0, y0, x1, y1 (exclusive) in interior coords
    baseline_rel_profile: Optional[int]
    solid_background: bool
    components: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _dominant_color(rgb: np.ndarray) -> np.ndarray:
    q = (rgb >> 4).astype(np.int32)
    key = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    counts = np.bincount(key.ravel(), minlength=4096)
    best = int(counts.argmax())
    sel = key == best
    return rgb[sel].reshape(-1, 3).mean(axis=0)


def _ring_zone(dim: np.ndarray) -> tuple[np.ndarray, dict]:
    """Depth of the red-frame anti-alias residue along each interior side.

    Residue is DIM ink present along most of a side, starting at (or very near) the
    border and continuing inwards.  Real lettering is bright and covers only part of
    a line, so it never satisfies this test.  Returns (zone mask, per-side depths).
    """
    h, w = dim.shape
    limit = max(1, min(h, w) // 3)

    def depth(profile: np.ndarray) -> int:
        last, start, gap = -1, None, 0
        for k in range(min(limit, len(profile))):
            if profile[k] >= 0.4:
                if start is None:
                    if k > 6:
                        break
                    start = k
                last, gap = k, 0
            elif start is not None:
                gap += 1
                if gap > 1:
                    break
            elif k > 6:
                break
        return 0 if start is None else last + 1

    t = depth(dim.mean(axis=1)[:limit])
    b = depth(dim.mean(axis=1)[::-1][:limit])
    l = depth(dim.mean(axis=0)[:limit])
    r = depth(dim.mean(axis=0)[::-1][:limit])
    zone = np.zeros((h, w), dtype=bool)
    if t:
        zone[:t + 1, :] = True
    if b:
        zone[h - b - 1:, :] = True
    if l:
        zone[:, :l + 1] = True
    if r:
        zone[:, w - r - 1:] = True
    return zone, {"top": t, "bottom": b, "left": l, "right": r}


def color_palette(crop_rgb: np.ndarray, n: int = 6) -> list[dict]:
    """Dominant colours (RGB + pixel fraction) - colours only, no image content."""
    flat = crop_rgb.reshape(-1, 3)
    q = (flat >> 5).astype(np.int32)
    key = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    counts = np.bincount(key, minlength=512)
    out: list[dict] = []
    for k in np.argsort(counts)[::-1][:n]:
        if counts[k] == 0:
            break
        out.append({"rgb": [int(round(v)) for v in flat[key == k].mean(axis=0)],
                    "fraction": round(float(counts[k]) / key.size, 4)})
    return out


def analyze_interior(
    crop_rgb: np.ndarray,
    cfg: Optional[MaskConfig] = None,
    target_span_rel: Optional[tuple[int, int]] = None,
) -> InteriorAnalysis:
    """Measure the text inside an interior crop and build the removal mask.

    ``target_span_rel`` (x0, x1) optionally limits the target to ink components
    whose horizontal centre lies in the span; everything else stays untouched.
    """
    cfg = cfg or MaskConfig()
    h, w = crop_rgb.shape[:2]
    pix = crop_rgb.astype(np.float32)
    bg0 = _dominant_color(crop_rgb)
    d0 = np.abs(pix - bg0).max(axis=2)
    base = d0 < cfg.noise_floor * 1.5
    bg = np.median(pix[base], axis=0) if base.any() else bg0
    d = np.abs(pix - bg).max(axis=2)
    cand = d >= cfg.noise_floor
    empty = np.zeros((h, w), dtype=bool)
    bg_i = tuple(int(round(v)) for v in bg)
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    default_fg = (0, 0, 0) if lum > 128 else (255, 255, 255)
    if cand.sum() < 4:
        return InteriorAnalysis(False, bg_i, default_fg, 0.0, np.zeros((h, w), np.float32),
                                empty, empty.copy(), None, None, True, [], ["no ink found"])

    p90 = float(np.percentile(d[cand], 90))
    hi = cand & (d >= p90)
    fg = np.median(pix[hi], axis=0)
    v = fg - bg
    contrast = float(np.linalg.norm(v))
    if contrast < cfg.min_contrast:
        return InteriorAnalysis(False, bg_i, default_fg, contrast, np.zeros((h, w), np.float32),
                                empty, empty.copy(), None, None, True, [],
                                [f"contrast {contrast:.1f} below {cfg.min_contrast}"])

    alpha = np.clip(((pix - bg) @ v) / float(v @ v), 0.0, 1.0).astype(np.float32)
    ink = (alpha >= cfg.ink_alpha_low) & (d >= cfg.noise_floor)
    # Frame residue (dim ink hugging the interior edge) must not be allowed to connect with the
    # lettering: letters whose tops/bottoms reach it would merge into one "border" component and
    # never be removed.  Residue is left UNTOUCHED; only bright, text-like pixels stay in play.
    dim = ink & (alpha < 0.6)
    zone, ring_depth = _ring_zone(dim)
    ink = ink & ~(zone & (alpha < 0.6))
    # Ignore the outer band of the interior: red-frame anti-alias residue lives there and
    # would otherwise form a ring that swallows (or is merged with) the real text.
    eb = int(min(cfg.edge_ignore_px, max(0, (min(h, w) - 1) // 8)))
    if eb > 0:
        ink[:eb, :] = False
        ink[h - eb:, :] = False
        ink[:, :eb] = False
        ink[:, w - eb:] = False

    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    removal = np.zeros((h, w), dtype=bool)
    excluded = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    notes: list[str] = []
    if any(ring_depth.values()):
        notes.append(f"frame residue left untouched (depth px): {ring_depth}")
    for i in range(1, n):
        x, y, cw, ch, area = (int(t) for t in stats[i])
        sel = lab == i
        left = int(sel[:, eb].sum()); right = int(sel[:, w - 1 - eb].sum())
        top = int(sel[eb, :].sum()); bottom = int(sel[h - 1 - eb, :].sum())
        touches = (left >= cfg.edge_contact_ratio * h or right >= cfg.edge_contact_ratio * h or
                   top >= cfg.edge_contact_ratio * w or bottom >= cfg.edge_contact_ratio * w)
        cx = x + cw / 2.0
        outside_span = target_span_rel is not None and not (target_span_rel[0] <= cx <= target_span_rel[1])
        status = "text"
        if touches:
            status = "kept: touches interior border (likely UI edge, not text)"
        elif outside_span:
            status = "kept: outside target span"
        comps.append({"bbox_rel": [x, y, x + cw, y + ch], "area": area, "status": status})
        if status == "text":
            removal |= sel
        else:
            excluded |= sel
            notes.append(f"component {comps[-1]['bbox_rel']} {status}")

    if not removal.any():
        return InteriorAnalysis(False, bg_i, tuple(int(round(t)) for t in fg), contrast, alpha,
                                removal, excluded, None, None, True, comps, notes + ["no target ink"])

    ys, xs = np.where(removal)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    # Faint anti-alias fringe directly touching the removed glyphs (coverage below the ink
    # threshold) - removing it avoids a ghost outline.  Only pixels adjacent to the glyphs.
    if cfg.fringe_alpha > 0:
        core = np.zeros((h, w), dtype=bool)
        core[eb:h - eb, eb:w - eb] = True
        core &= ~zone
        adj = cv2.dilate(removal.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        removal = removal | (adj & core & ~excluded & (alpha >= cfg.fringe_alpha) & (d >= cfg.noise_floor * 0.5))

    near = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    bgpix = pix[~near]
    solid = bool(bgpix.size == 0 or float(bgpix.std(axis=0).max()) <= cfg.solid_bg_std)

    # profile-based baseline (fallback when the original text is unknown)
    prof = (alpha * removal).sum(axis=1)[bbox[1]:bbox[3]]
    baseline = None
    if prof.size >= 4:
        drops = prof[:-1] - prof[1:]
        lo = int(len(drops) * 0.4)
        r = int(np.argmax(drops[lo:])) + lo
        baseline = bbox[1] + r + 1  # first row below the body of the text

    # The old frame's soft edge inside the interior (dim strips along the border) is annotation residue, not
    # content: remove it with the old text so no faint box is left behind.  Position/baseline above were measured
    # from the lettering only, so this does not move the replacement.
    if cfg.remove_frame_residue and any(ring_depth.values()):
        residue = cv2.dilate((zone & dim).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & zone
        removal = removal | (residue & ~excluded)
        notes.append(f"frame residue removed (depth px): {ring_depth}")

    return InteriorAnalysis(True, bg_i, tuple(int(round(t)) for t in fg), contrast, alpha,
                            removal, excluded, bbox, baseline, solid, comps, notes)


def _smooth_fill(crop_rgb: np.ndarray, fill_mask: np.ndarray, invalid: np.ndarray,
                 thickness: float, fallback_rgb: tuple) -> np.ndarray:
    """Estimate the background under ``fill_mask`` from surrounding clean pixels
    (normalised Gaussian convolution; scale follows the stroke thickness)."""
    pix = crop_rgb.astype(np.float32)
    valid = (~invalid).astype(np.float32)
    est = np.zeros_like(pix)
    done = np.zeros(fill_mask.shape, dtype=bool)
    sigma = float(np.clip(thickness * 0.75, 3.0, 40.0))
    num_src = pix * valid[..., None]
    for _ in range(5):
        den = cv2.GaussianBlur(valid, (0, 0), sigma)
        num = np.stack([cv2.GaussianBlur(num_src[..., c], (0, 0), sigma) for c in range(3)], axis=2)
        ok = (den > 1e-3) & ~done
        est[ok] = num[ok] / den[ok][:, None]
        done |= ok
        if done[fill_mask].all():
            break
        sigma *= 2.0
    est[~done] = np.array(fallback_rgb, dtype=np.float32)
    return np.clip(np.rint(est), 0, 255).astype(np.uint8)


def reconstruct_background(crop_rgb: np.ndarray, an: InteriorAnalysis, cfg: Optional[MaskConfig] = None) -> np.ndarray:
    """Replace ONLY the removal-mask pixels with reconstructed background."""
    cfg = cfg or MaskConfig()
    out = crop_rgb.copy()
    rm = an.removal_mask
    if not rm.any():
        return out
    if an.solid_background:
        out[rm] = np.array(an.bg_rgb, dtype=np.uint8)
    else:
        # Thick glyphs defeat small-radius inpainting (it leaves ghost blobs shaped like the
        # old text), so use a smooth fill whose scale follows the stroke thickness.
        dt = cv2.distanceTransform(rm.astype(np.uint8), cv2.DIST_L2, 3)
        thickness = 2.0 * float(dt.max())
        guard = cv2.dilate((rm | an.excluded_mask).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        fill = _smooth_fill(crop_rgb, rm, guard, thickness, an.bg_rgb)
        out[rm] = fill[rm]
    return out


# =============================================================================
# TEXT RENDERING
# =============================================================================

_CMAP_CACHE: dict[str, set[int]] = {}


def font_codepoints(font_path: str) -> Optional[set[int]]:
    """Set of code points a font can render (None if fontTools is unavailable)."""
    if TTFont is None:
        return None
    key = str(font_path)
    if key not in _CMAP_CACHE:
        f = TTFont(key, lazy=True, fontNumber=0)
        try:
            _CMAP_CACHE[key] = set(f.getBestCmap().keys())
        finally:
            f.close()
    return _CMAP_CACHE[key]


def load_font(font_path: str, size_px: int) -> ImageFont.FreeTypeFont:
    p = Path(font_path)
    if not p.is_file():
        raise FileNotFoundError(f"Font file not found: {font_path}")
    try:
        return ImageFont.truetype(str(p), size=int(size_px))
    except OSError as exc:
        raise OSError(f"Font file could not be loaded: {font_path} ({exc})") from exc


def check_glyphs(text: str, font_path: str) -> list[str]:
    """Characters of ``text`` the font has no glyph for (never silently drawn as .notdef)."""
    cps = font_codepoints(font_path)
    if cps is None:
        return []
    return sorted({ch for ch in text if ch not in ("\n", "\r") and ord(ch) not in cps and not ch.isspace()})


def layout_positions(text: str, font: ImageFont.FreeTypeFont, size_px: float,
                     tracking_em: float = 0.0, word_spacing_em: float = 0.0) -> tuple[list[float], float]:
    """x origin of every glyph plus total advance, with pair kerning and tracking."""
    xs: list[float] = []
    x = 0.0
    for i, ch in enumerate(text):
        xs.append(x)
        adv = font.getlength(ch)
        kern = 0.0
        if i + 1 < len(text):
            nxt = text[i + 1]
            kern = font.getlength(ch + nxt) - adv - font.getlength(nxt)
        x += adv + kern + tracking_em * size_px + (word_spacing_em * size_px if ch == " " else 0.0)
    return xs, x


def render_text_alpha(text: str, font_path: str, size_px: int, canvas_wh: tuple[int, int],
                      origin_xy: tuple[float, float], tracking_em: float = 0.0,
                      word_spacing_em: float = 0.0, stroke_width: int = 0) -> np.ndarray:
    """Antialiased coverage layer (uint8) of ``text``; origin = left of first glyph, BASELINE y."""
    font = load_font(font_path, size_px)
    W, H = canvas_wh
    layer = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(layer)
    xs, _ = layout_positions(text, font, size_px, tracking_em, word_spacing_em)
    ox, oy = origin_xy
    for ch, gx in zip(text, xs):
        if ch.isspace():
            continue
        draw.text((ox + gx, oy), ch, font=font, fill=255, anchor="ls", stroke_width=stroke_width)
    return np.asarray(layer)


def alpha_bbox(alpha: np.ndarray, thr: int = 0) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(alpha > thr)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def render_scratch(text: str, font_path: str, size_px: int, tracking_em: float = 0.0,
                   word_spacing_em: float = 0.0, stroke_width: int = 0) -> tuple[np.ndarray, tuple[int, int]]:
    """Render on a generous canvas; returns (alpha, origin) with origin=(x, baseline)."""
    font = load_font(font_path, size_px)
    _, adv = layout_positions(text, font, size_px, tracking_em, word_spacing_em)
    pad = int(size_px * 1.5) + 2 * stroke_width + 4
    W = int(math.ceil(max(adv, 1))) + 2 * pad
    H = int(size_px * 3) + 2 * pad
    origin = (pad, pad + int(size_px * 1.5))
    return render_text_alpha(text, font_path, size_px, (W, H), origin, tracking_em,
                             word_spacing_em, stroke_width), origin


def font_baseline_below(text: str, font_path: str, size_px: int, stroke_width: int = 0) -> int:
    """How many pixels the ink of ``text`` extends BELOW the baseline (descenders)."""
    a, origin = render_scratch(text, font_path, size_px, 0.0, 0.0, stroke_width)
    bb = alpha_bbox(a)
    return 0 if bb is None else max(0, bb[3] - origin[1])


# =============================================================================
# REPLACEMENT PLANNING
# =============================================================================

@dataclass
class ReplacementPlan:
    text: str
    font_path: str
    size_px: int
    tracking_em: float
    stroke_width: int
    color_rgb: tuple[int, int, int]
    alignment: str
    baseline_rel: int
    alpha: np.ndarray                 # uint8 (h, w) clipped to the interior
    ink_bbox_rel: tuple[int, int, int, int]
    font_used: str
    notes: list[str] = field(default_factory=list)


def decide_alignment(an: InteriorAnalysis, inner_w: int) -> str:
    """left / center / right from the ORIGINAL ink position inside the interior."""
    if an.ink_bbox_rel is None:
        return "center"
    x0, _, x1, _ = an.ink_bbox_rel
    left_gap, right_gap = x0, inner_w - x1
    tol = max(5, int(0.03 * inner_w))
    if abs(left_gap - right_gap) <= tol:
        return "center"
    return "left" if left_gap < right_gap else "right"


def _try_place(text: str, typo: dict, an: InteriorAnalysis, inner_wh: tuple[int, int],
               tracking_em: float, cfg: MaskConfig, clamp: bool = False) -> ReplacementPlan:
    """Lay the text out at the original text's anchor/baseline or raise PlacementError."""
    w, h = inner_wh
    font_path = typo["font_path"]
    size = int(typo["font_size"])
    stroke = int(typo.get("stroke_width", 0))
    word_sp = float(typo.get("word_spacing_em", 0.0))
    missing = check_glyphs(text, font_path)
    if missing:
        raise PlacementError(f"font {Path(font_path).name} has no glyph for: {''.join(missing)!r}")

    scratch, (ox_s, oy_s) = render_scratch(text, font_path, size, tracking_em, word_sp, stroke)
    bb = alpha_bbox(scratch)
    if bb is None:
        raise PlacementError("replacement text renders no visible pixels")
    ink_w = bb[2] - bb[0]

    # horizontal anchor: the ORIGINAL ink edge / centre, not the interior centre
    alignment = typo.get("alignment") or decide_alignment(an, w)
    ox0, ox1 = (an.ink_bbox_rel[0], an.ink_bbox_rel[2]) if an.ink_bbox_rel is not None else (0, w)
    if alignment == "left":
        target_x0 = ox0
    elif alignment == "right":
        target_x0 = ox1 - ink_w
    else:
        target_x0 = int(round((ox0 + ox1) / 2.0 - ink_w / 2.0))

    # vertical anchor: the baseline (calibrated > profile estimate > ink bottom)
    baseline = typo.get("baseline_rel")
    if baseline is None:
        if an.baseline_rel_profile is not None:
            baseline = an.baseline_rel_profile
        elif an.ink_bbox_rel is not None:
            baseline = an.ink_bbox_rel[3]
        else:
            baseline = int(0.7 * h)
    baseline = int(round(baseline))

    if typo.get("anchor") == "interior_center":      # monogram / short label in a square tile
        target_x0 = int(round((w - ink_w) / 2.0))
        top_rel, bottom_rel = bb[1] - oy_s, bb[3] - oy_s     # ink extent relative to the baseline
        baseline = int(round(h / 2.0 - (top_rel + bottom_rel) / 2.0))

    dx = target_x0 - bb[0]                      # scratch x -> interior x
    ys, xs = np.where(scratch > cfg.overflow_alpha_tolerance)
    if xs.size:
        sx0, sx1 = int(xs.min()) + dx, int(xs.max()) + 1 + dx
        sy0 = int(ys.min()) - oy_s + baseline
        sy1 = int(ys.max()) + 1 - oy_s + baseline
        if clamp:   # keep the font size; only slide the text back inside the interior if it is small enough
            if sx1 - sx0 <= w:
                if sx0 < 0:
                    dx -= sx0; sx1 -= sx0; sx0 = 0
                if sx1 > w:
                    dx -= sx1 - w; sx0 -= sx1 - w; sx1 = w
            if sy1 - sy0 <= h:
                if sy0 < 0:
                    baseline -= sy0; sy1 -= sy0; sy0 = 0
                if sy1 > h:
                    baseline -= sy1 - h; sy0 -= sy1 - h; sy1 = h
        if sx0 < 0 or sx1 > w:
            raise PlacementError(
                f"replacement is {ink_w}px wide at this size; ink would span x={sx0}..{sx1} "
                f"but the interior is 0..{w}")
        if sy0 < 0 or sy1 > h:
            raise PlacementError(
                f"replacement ink would span y={sy0}..{sy1} but the interior is 0..{h} tall")

    final = render_text_alpha(text, font_path, size, (w, h), (float(ox_s + dx), float(baseline)),
                              tracking_em, word_sp, stroke)
    fbb = alpha_bbox(final)
    if fbb is None:
        raise PlacementError("replacement is entirely outside the interior")
    color = tuple(int(c) for c in (typo.get("text_color_rgb") or an.fg_rgb))
    return ReplacementPlan(text, font_path, size, tracking_em, stroke, color, alignment, baseline,
                           final, fbb, Path(font_path).name)


def plan_replacement(text: str, an: InteriorAnalysis, inner_wh: tuple[int, int],
                     typo: dict, cfg: Optional[MaskConfig] = None) -> ReplacementPlan:
    """Plan the replacement without touching any pixels.

    Overflow handling, in order:
      1. calibrated font/size/tracking
      2. slightly tighter tracking (at most ``max_tracking_tighten_em`` below calibrated)
      3. alternative fonts that scored within ``alt_font_score_margin`` of the best
      otherwise PlacementError - the text is never shrunk, stretched or clipped.
    """
    cfg = cfg or MaskConfig()
    text = " ".join(str(text).split())
    if not text:
        raise PlacementError("empty replacement text")
    attempts: list[tuple[dict, float, str, bool]] = []
    base_track = float(typo.get("tracking_em", 0.0))
    tight_track = base_track - cfg.max_tracking_tighten_em
    attempts.append((typo, base_track, "calibrated typography", False))
    attempts.append((typo, tight_track, "tracking tightened", False))
    for alt in typo.get("alternatives", []) or []:
        merged = dict(typo)
        merged.update({k: alt[k] for k in ("font_path", "font_size", "stroke_width") if k in alt})
        attempts.append((merged, float(alt.get("tracking_em", base_track)), f"alternative font {Path(alt['font_path']).name}", False))
    if cfg.allow_clamp:
        attempts.append((typo, base_track, "slid back inside the box (same size)", True))
        attempts.append((typo, tight_track, "tracking tightened + slid back inside the box", True))

    last_err: Optional[Exception] = None
    for tp, trk, label, clamp in attempts:
        try:
            plan = _try_place(text, tp, an, inner_wh, trk, cfg, clamp)
            plan.notes.append(label)
            return plan
        except PlacementError as exc:
            last_err = exc
    raise PlacementError(str(last_err))


# =============================================================================
# EDIT EXECUTION
# =============================================================================

@dataclass
class EditResult:
    removal_mask: np.ndarray       # bool (h, w) relative to interior
    replacement_mask: np.ndarray   # bool (h, w)
    allowed_mask: np.ndarray       # bool (h, w) = removal | replacement
    inner_rect: tuple[int, int, int, int]
    residual_ink_pixels: int = 0   # old-text-like ink still present after removal (before new text)


def execute_edit(img_rgb: np.ndarray, inner_rect: tuple[int, int, int, int], an: InteriorAnalysis,
                 plan: ReplacementPlan, cfg: Optional[MaskConfig] = None) -> EditResult:
    """Apply one replacement IN PLACE on ``img_rgb`` (only inside the interior)."""
    cfg = cfg or MaskConfig()
    x, y, w, h = inner_rect
    crop = img_rgb[y:y + h, x:x + w]
    recon = reconstruct_background(crop, an, cfg)
    an2 = analyze_interior(recon, cfg)
    residual = int(an2.removal_mask.sum()) if an2.found else 0
    a = plan.alpha.astype(np.float32) / 255.0
    rep = plan.alpha > 0
    color = np.array(plan.color_rgb, dtype=np.float32)
    comp = recon.astype(np.float32) * (1.0 - a[..., None]) + color * a[..., None]
    new = recon.copy()
    new[rep] = np.clip(np.rint(comp[rep]), 0, 255).astype(np.uint8)
    img_rgb[y:y + h, x:x + w] = new
    return EditResult(an.removal_mask.copy(), rep, an.removal_mask | rep, inner_rect, residual)


def remove_only(img_rgb: np.ndarray, inner_rect: tuple[int, int, int, int], an: InteriorAnalysis,
                cfg: Optional[MaskConfig] = None) -> EditResult:
    """Erase the original text without drawing anything (empty replacement)."""
    x, y, w, h = inner_rect
    img_rgb[y:y + h, x:x + w] = reconstruct_background(img_rgb[y:y + h, x:x + w], an, cfg)
    z = np.zeros_like(an.removal_mask)
    return EditResult(an.removal_mask.copy(), z, an.removal_mask.copy(), inner_rect)


# =============================================================================
# PRESERVATION VALIDATION
# =============================================================================

def build_allowed_mask(shape_hw: tuple[int, int], annotation_mask: Optional[np.ndarray],
                       edits: list[EditResult]) -> np.ndarray:
    """Union of ONLY the explicitly authorised pixels (never whole boxes)."""
    allowed = np.zeros(shape_hw, dtype=bool)
    if annotation_mask is not None:
        allowed |= annotation_mask
    for e in edits:
        x, y, w, h = e.inner_rect
        allowed[y:y + h, x:x + w] |= e.allowed_mask
    return allowed


def validate_preservation(original: np.ndarray, output: np.ndarray, allowed: np.ndarray,
                          raise_on_fail: bool = True) -> dict:
    """Compare original vs output; every changed pixel must lie inside ``allowed``."""
    if original.shape != output.shape or original.dtype != output.dtype:
        raise PreservationError(
            f"shape/dtype changed: {original.shape} {original.dtype} -> {output.shape} {output.dtype}")
    if allowed.shape != original.shape[:2] or allowed.dtype != bool:
        raise PreservationError("allowed mask must be a boolean array matching the image")
    changed = (original != output).reshape(original.shape[0], original.shape[1], -1).any(axis=2)
    bad = changed & ~allowed
    n_bad = int(bad.sum())
    result = {
        "passed": n_bad == 0,
        "changed_pixels": int(changed.sum()),
        "allowed_pixels": int(allowed.sum()),
        "unauthorized_changed_pixels": n_bad,
        "image_pixels": int(allowed.size),
        "outside_mask_identical": bool(np.array_equal(original[~allowed], output[~allowed])),
    }
    if n_bad:
        ys, xs = np.where(bad)
        result["first_unauthorized_pixels_xy"] = [[int(a), int(b)] for a, b in zip(xs[:10], ys[:10])]
        if raise_on_fail:
            raise PreservationError(
                f"{n_bad} pixel(s) changed outside the authorised mask, "
                f"first at {result['first_unauthorized_pixels_xy'][:3]}")
    return result


def verify_within_boxes(allowed_by_box: list[tuple[dict, np.ndarray]], shape_hw: tuple[int, int]) -> list[str]:
    """Each box's edit mask must lie inside that box's interior and outer red box."""
    problems: list[str] = []
    for box, mask_full in allowed_by_box:
        ix, iy, iw, ih = inner_rect_of(box)
        inside = np.zeros(shape_hw, dtype=bool)
        inside[iy:iy + ih, ix:ix + iw] = True
        if (mask_full & ~inside).any():
            problems.append(f"edit for box at ({box['x']},{box['y']}) escapes its interior")
        ox, oy, ow, oh = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        outer = np.zeros(shape_hw, dtype=bool)
        outer[oy:oy + oh, ox:ox + ow] = True
        if (mask_full & ~outer).any():
            problems.append(f"edit for box at ({box['x']},{box['y']}) escapes its outer red box")
    return problems


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def write_diagnostics(out_dir: Path, original_rgb: np.ndarray, output_rgb: np.ndarray, allowed: np.ndarray,
                      boxes: list[dict], per_box: list[dict], report: dict) -> None:
    """Diagnostic files only - never the final image.  Contains crops of the source image."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_rgb(name: str, arr: np.ndarray) -> None:
        cv2.imwrite(str(out_dir / name), np.ascontiguousarray(arr[..., ::-1]))

    changed = (original_rgb != output_rgb).any(axis=2)
    save_rgb("diag_allowed_mask.png", np.repeat((allowed * 255).astype(np.uint8)[..., None], 3, axis=2))
    vis = original_rgb.copy()
    vis[changed & allowed] = (0, 200, 0)
    vis[changed & ~allowed] = (255, 0, 0)
    save_rgb("diag_changed_pixels.png", vis)
    diff = np.abs(original_rgb.astype(np.int16) - output_rgb.astype(np.int16)).max(axis=2)
    save_rgb("diag_difference_x8.png", np.repeat(np.clip(diff * 8, 0, 255).astype(np.uint8)[..., None], 3, axis=2))
    for i, (box, info) in enumerate(zip(boxes, per_box), start=1):
        ox, oy, ow, oh = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        pad = 12
        y0, y1 = max(0, oy - pad), min(original_rgb.shape[0], oy + oh + pad)
        x0, x1 = max(0, ox - pad), min(original_rgb.shape[1], ox + ow + pad)
        before, after = original_rgb[y0:y1, x0:x1], output_rgb[y0:y1, x0:x1]
        sbs = np.concatenate([before, np.full((before.shape[0], 8, 3), 128, np.uint8), after], axis=1)
        save_rgb(f"diag_box{i}_before_after.png", sbs)
        if "removal_mask" in info and info["removal_mask"] is not None:
            ix, iy, iw, ih = inner_rect_of(box)
            for key in ("removal_mask", "replacement_mask", "allowed_mask"):
                m = info.get(key)
                if m is not None:
                    cv2.imwrite(str(out_dir / f"diag_box{i}_{key}.png"), np.ascontiguousarray(m.astype(np.uint8) * 255))
    clean = json.loads(json.dumps(report, default=_json_default))
    (out_dir / "replacement_report.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return None
    if isinstance(o, Path):
        return str(o)
    return str(o)
