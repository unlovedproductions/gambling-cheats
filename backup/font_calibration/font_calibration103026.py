#!/usr/bin/env python3
"""Per-box font calibration for text_replacer_complete.py.

For every box this module

1. measures the ORIGINAL text from the image (tight ink mask, antialiased
   coverage, ink bounds, foreground colour, stroke thickness) - the interior
   analysis is shared with text_masks.py;
2. discovers installed .ttf/.otf fonts and reads their metadata with fontTools;
3. ranks candidate fonts by *measurable* evidence at matched glyph height:
   total width, column profile, row profile (x-height / cap / descender
   structure), stroke thickness and a distance-transform contour (chamfer)
   comparison - weights are in ``RANKING_WEIGHTS``;
4. only AFTER the font is chosen, fits size and letter spacing separately
   (font identity is never decided by stretching spacing to match width);
5. derives the baseline from the font's own descender extent for the OCR text.

Everything here is an image-based estimate.  A font that cannot be told apart
from another (e.g. Liberation Sans vs. Arimo, which share outlines) is
reported as ambiguous rather than declared the original.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import text_masks as tm

try:
    from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover
    TTFont = None


# =============================================================================
# CONFIGURATION
# =============================================================================

RANKING_WEIGHTS = {
    "width": 0.25,      # total ink width at matched height (no spacing/scale tricks)
    "columns": 0.20,    # horizontal ink profile (glyph widths, counters, spacing pattern)
    "rows": 0.20,       # vertical ink profile (x-height / cap-height / descender structure)
    "stroke": 0.15,     # stroke thickness
    "chamfer": 0.20,    # distance-transform contour similarity
}
AMBIGUOUS_MARGIN = 0.02      # best vs runner-up closer than this -> "ambiguous"
LOW_CONFIDENCE_MARGIN = 0.06
MAX_CANDIDATES = 150
MAX_FONT_BYTES = 6 * 1024 * 1024
TRACKING_LIMITS = (-0.08, 0.15)
TRACKING_SUSPECT_EM = 0.06   # |tracking| above this -> font identity / OCR text doubtful
BOLD_HINT_PRIOR = 0.03       # soft prior for boxes the user knows are bold (legacy: box 2)
SYNTHETIC_BOLD_MIN_GAIN = 0.03

PRIORITY_NAMES = ("arial", "helvetica", "liberation sans", "liberationsans", "nimbus sans", "arimo")
SECONDARY_PRIORITY_NAMES = ("roboto", "dejavu sans", "dejavusans", "noto sans", "open sans", "inter")
EXCLUDE_TOKENS = ("mono", "code", "console", "courier", "symbol", "emoji", "dingbat", "wingding",
                  "math", "script", "hand", "slab", "cjk", "gothic", "mincho", "icon", "barcode")


def default_font_dirs(script_dir: Path) -> list[Path]:
    """Project directories first, then per-OS system font directories."""
    dirs = [script_dir, script_dir / "fonts"]
    home = Path.home()
    system = platform.system()
    if system == "Windows":
        windir = Path(os.environ.get("WINDIR", "C:/Windows"))
        dirs += [windir / "Fonts", Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local")) / "Microsoft/Windows/Fonts"]
    elif system == "Darwin":
        dirs += [Path("/Library/Fonts"), Path("/System/Library/Fonts"), home / "Library/Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".fonts", home / ".local/share/fonts"]
    return dirs


# =============================================================================
# FONT DISCOVERY / METADATA
# =============================================================================

@dataclass
class FontInfo:
    path: str
    family: str = ""
    subfamily: str = ""
    full_name: str = ""
    postscript_name: str = ""
    version: str = ""
    weight_class: int = 0
    width_class: int = 0
    is_italic: bool = False
    is_bold: bool = False
    units_per_em: int = 0
    ascender: int = 0
    descender: int = 0
    cap_height: int = 0
    x_height: int = 0
    glyph_count: int = 0
    priority: int = 2
    valid: bool = True
    error: str = ""

    def label(self) -> str:
        return f"{self.family} {self.subfamily}".strip() or Path(self.path).stem


def font_metadata(path: Path) -> FontInfo:
    """Read metadata with fontTools; never raises (invalid fonts get valid=False)."""
    info = FontInfo(path=str(path))
    if TTFont is None:
        info.valid, info.error = False, "fontTools not installed"
        return info
    try:
        f = TTFont(str(path), lazy=True, fontNumber=0)
    except Exception as exc:
        info.valid, info.error = False, f"{type(exc).__name__}: {exc}"
        return info
    try:
        name = f["name"]

        def nm(i: int) -> str:
            rec = name.getName(i, 3, 1, 0x409) or name.getName(i, 1, 0, 0)
            return rec.toUnicode() if rec else ""

        info.family = nm(16) or nm(1)
        info.subfamily = nm(17) or nm(2)
        info.full_name, info.postscript_name, info.version = nm(4), nm(6), nm(5)
        os2 = f["OS/2"] if "OS/2" in f else None
        head = f["head"]
        if os2 is not None:
            info.weight_class = int(getattr(os2, "usWeightClass", 0))
            info.width_class = int(getattr(os2, "usWidthClass", 0))
            fs = int(getattr(os2, "fsSelection", 0))
            info.is_italic = bool(fs & 1) or bool(head.macStyle & 2)
            info.is_bold = bool(fs & 32) or info.weight_class >= 600
            info.cap_height = int(getattr(os2, "sCapHeight", 0) or 0)
            info.x_height = int(getattr(os2, "sxHeight", 0) or 0)
        else:
            info.is_italic = bool(head.macStyle & 2)
            info.is_bold = bool(head.macStyle & 1)
        info.units_per_em = int(head.unitsPerEm)
        if "hhea" in f:
            info.ascender, info.descender = int(f["hhea"].ascent), int(f["hhea"].descent)
        info.glyph_count = int(f["maxp"].numGlyphs)
    except Exception as exc:
        info.valid, info.error = False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            f.close()
        except Exception:
            pass
    lname = f"{info.family} {info.full_name} {Path(path).stem}".lower()
    if any(t in lname for t in PRIORITY_NAMES):
        info.priority = 0
    elif any(t in lname for t in SECONDARY_PRIORITY_NAMES):
        info.priority = 1
    return info


def discover_fonts(search_dirs: list[Path], max_bytes: int = MAX_FONT_BYTES) -> tuple[list[FontInfo], list[FontInfo]]:
    """Recursively find .ttf/.otf files. Returns (valid fonts, invalid/skipped fonts)."""
    seen: set[str] = set()
    valid: list[FontInfo] = []
    invalid: list[FontInfo] = []
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in (".ttf", ".otf") or not p.is_file():
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            info = font_metadata(p)
            (valid if info.valid else invalid).append(info)
    valid.sort(key=lambda i: (i.priority, 0 if i.path.lower().endswith(".ttf") else 1, i.path.lower()))
    return valid, invalid


def is_eligible(info: FontInfo, text: str) -> bool:
    """Sans-serif, upright, non-symbol fonts that can render every character of ``text``."""
    if not info.valid or info.is_italic:
        return False
    lname = f"{info.family} {info.subfamily} {Path(info.path).stem}".lower()
    if any(t in lname for t in EXCLUDE_TOKENS):
        return False
    if "serif" in lname and "sans" not in lname:
        return False
    cps = tm.font_codepoints(info.path)
    if cps is not None and any(ord(c) not in cps for c in text if not c.isspace()):
        return False
    return True


# =============================================================================
# FEATURES / SCORING
# =============================================================================

def _features(alpha01: np.ndarray) -> Optional[dict[str, Any]]:
    """Shape descriptors of an ink coverage map already cropped to its bounding box."""
    if alpha01.size == 0 or alpha01.sum() <= 0:
        return None
    binary = (alpha01 >= 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours))
    area = float(alpha01.sum())
    return {
        "h": alpha01.shape[0], "w": alpha01.shape[1], "area": area,
        "stroke": (2.0 * area / perim) if perim > 0 else 0.0,
        "cols": alpha01.sum(axis=0), "rows": alpha01.sum(axis=1), "binary": binary,
    }


def _ncc_error(a: np.ndarray, b: np.ndarray) -> float:
    """(1 - Pearson r) / 2 after resampling b to len(a); resampling is used for COMPARISON only."""
    if len(a) < 3 or len(b) < 3:
        return 1.0
    b2 = np.interp(np.linspace(0, len(b) - 1, len(a)), np.arange(len(b)), b)
    if a.std() < 1e-6 or b2.std() < 1e-6:
        return 1.0
    return float((1.0 - np.corrcoef(a, b2)[0, 1]) / 2.0)


def _chamfer_error(orig_bin: np.ndarray, cand_alpha: np.ndarray) -> float:
    h, w = orig_bin.shape
    cand = cv2.resize(cand_alpha.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5
    o = orig_bin.astype(bool)
    if not cand.any() or not o.any():
        return 1.0
    d_o = cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3)
    d_c = cv2.distanceTransform((~cand).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = (float(d_o[cand].mean()) + float(d_c[o].mean())) / 2.0
    return float(min(1.0, (chamfer / max(h, 1)) / 0.08))


def _render_features(text: str, font_path: str, size: int, stroke: int,
                     tracking: float = 0.0) -> Optional[tuple[dict[str, Any], np.ndarray, tuple[int, int, int, int], tuple[int, int]]]:
    a, origin = tm.render_scratch(text, font_path, size, tracking, 0.0, stroke)
    bb = tm.alpha_bbox(a)
    if bb is None:
        return None
    crop = a[bb[1]:bb[3], bb[0]:bb[2]].astype(np.float32) / 255.0
    f = _features(crop)
    return (f, crop, bb, origin) if f else None


def solve_size(text: str, font_path: str, target_h: int, stroke: int = 0) -> Optional[int]:
    """Integer size whose rendered ink height best matches ``target_h`` (linear scaling + local refine)."""
    probe = _render_features(text, font_path, 100, stroke)
    if probe is None:
        return None
    h100 = probe[0]["h"]
    size = max(6, int(round(100.0 * target_h / max(h100 - 2 * stroke, 1))))
    best, best_err = None, 1e9
    for _ in range(6):
        r = _render_features(text, font_path, size, stroke)
        if r is None:
            return None
        err = r[0]["h"] - target_h
        if abs(err) < best_err:
            best, best_err = size, abs(err)
        if err == 0:
            break
        size = max(6, size - (1 if err > 0 else -1))
    return best


def score_candidate(text: str, orig: dict[str, Any], font_path: str, size: int, stroke: int) -> Optional[dict[str, Any]]:
    r = _render_features(text, font_path, size, stroke)
    if r is None:
        return None
    f, crop, _, _ = r
    err = {
        "width": min(1.0, abs(f["w"] - orig["w"]) / max(orig["w"], 1) / 0.25),
        "columns": _ncc_error(orig["cols"], f["cols"]),
        "rows": _ncc_error(orig["rows"], f["rows"]),
        "stroke": min(1.0, abs(f["stroke"] - orig["stroke"]) / max(orig["stroke"], 1e-3) / 0.5),
        "chamfer": _chamfer_error(orig["binary"], crop),
    }
    score = sum(RANKING_WEIGHTS[k] * err[k] for k in RANKING_WEIGHTS)
    return {"score": float(score), "errors": {k: round(float(v), 4) for k, v in err.items()},
            "rendered_w": f["w"], "rendered_h": f["h"], "rendered_stroke": round(f["stroke"], 3)}


def rank_fonts(text: str, orig_alpha01: np.ndarray, fonts: list[FontInfo], bold_hint: bool = False,
               max_candidates: int = MAX_CANDIDATES) -> list[dict[str, Any]]:
    """Rank eligible fonts for ``text`` against the original ink coverage map."""
    orig = _features(orig_alpha01)
    if orig is None or not text.strip():
        return []
    elig = [f for f in fonts if is_eligible(f, text)]
    elig.sort(key=lambda i: (i.priority, i.path.lower()))
    elig = elig[:max_candidates]
    rows: list[dict[str, Any]] = []
    for info in elig:
        try:
            size = solve_size(text, info.path, orig["h"], 0)
            if size is None:
                continue
            s = score_candidate(text, orig, info.path, size, 0)
        except (OSError, ValueError):
            continue
        if s is None:
            continue
        prior = -BOLD_HINT_PRIOR if (bold_hint and info.is_bold) else 0.0
        rows.append({"path": info.path, "label": info.label(), "weight_class": info.weight_class,
                     "width_class": info.width_class, "is_bold": info.is_bold, "priority": info.priority,
                     "size_px": size, "stroke_width": 0, "synthetic_bold": False,
                     "score": s["score"] + prior, "raw_score": s["score"], "errors": s["errors"],
                     "rendered_w": s["rendered_w"], "rendered_h": s["rendered_h"]})
    rows.sort(key=lambda r: (round(r["score"], 6), r["priority"], r["path"].lower()))

    # synthetic bold only if the original is clearly heavier than the best regular candidates
    if rows:
        top = rows[0]
        rr = _render_features(text, top["path"], top["size_px"], 0)
        if rr and orig["stroke"] / max(rr[0]["stroke"], 1e-3) > 1.3:
            for base in [r for r in rows[:3] if not r["is_bold"]]:
                for sw in (1, 2):
                    try:
                        sz = solve_size(text, base["path"], orig["h"], sw)
                        s = score_candidate(text, orig, base["path"], sz, sw) if sz else None
                    except (OSError, ValueError):
                        s = None
                    if s and s["score"] + SYNTHETIC_BOLD_MIN_GAIN < rows[0]["score"]:
                        rows.append({**base, "size_px": sz, "stroke_width": sw, "synthetic_bold": True,
                                     "score": s["score"], "raw_score": s["score"], "errors": s["errors"],
                                     "rendered_w": s["rendered_w"], "rendered_h": s["rendered_h"],
                                     "label": base["label"] + f" +synthetic stroke {sw}"})
            rows.sort(key=lambda r: (round(r["score"], 6), r["priority"], r["path"].lower()))
    return rows


def _confidence(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if not rows:
        return "fallback", []
    if len(rows) == 1:
        return "estimated", []
    margin = rows[1]["score"] - rows[0]["score"]
    near = [r["label"] for r in rows[1:6] if r["score"] - rows[0]["score"] < AMBIGUOUS_MARGIN]
    if margin < AMBIGUOUS_MARGIN:
        return "ambiguous", near
    if margin < LOW_CONFIDENCE_MARGIN:
        return "estimated-low", near
    return "estimated", []


# =============================================================================
# LEGACY HELPERS (kept for compatibility)
# =============================================================================

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
    """Find non-background, non-red ink pixels in a box crop (legacy bounding-box measurement)."""
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
    color = np.median(pixels[mask], axis=0).clip(0, 255).astype(int).tolist()
    return {"found": True, "width": x1 - x0, "height": y1 - y0, "bbox": [x0, y0, x1, y1], "color_rgb": color}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def calibrate_font_parameters(
    image_rgb: Image.Image,
    boxes: list[dict],
    original_texts: list[str],
    default_font_path: str,
    background_rgb: tuple[int, int, int],
    script_dir: Path,
    analyses: Optional[list[tm.InteriorAnalysis]] = None,
    font_dirs: Optional[list[Path]] = None,
    bold_box_indices: tuple[int, ...] = (2,),
    report_path: Optional[Path] = None,
    mask_config: Optional[tm.MaskConfig] = None,
    verbose: bool = True,
    family_allowlist: Optional[tuple[str, ...]] = None,
) -> list[dict[str, Any]]:
    """Run calibration before replacement and return one config per box.

    ``boxes`` may be the legacy ``{"x","y","w","h"}`` dicts or the new dicts with
    an ``"inner"`` rectangle (full-image coordinates).  ``bold_box_indices`` is a
    soft prior for boxes known to be bold (legacy: box 2); the measured stroke
    thickness still decides.
    """
    cfg = mask_config or tm.MaskConfig()
    script_dir = Path(script_dir)
    dirs = font_dirs if font_dirs is not None else default_font_dirs(script_dir)
    fonts, invalid = discover_fonts(dirs)
    # Fonts in the project's fonts/ folder (including downloaded ones) must never be
    # cut off by the MAX_CANDIDATES limit when the system has hundreds of fonts.
    project_fonts = (script_dir / "fonts").resolve()
    for f in fonts:
        try:
            if project_fonts in Path(f.path).resolve().parents:
                f.priority = min(f.priority, 1)
        except OSError:
            pass
    fonts.sort(key=lambda i: (i.priority, 0 if i.path.lower().endswith(".ttf") else 1, i.path.lower()))
    if family_allowlist:
        deny = ("narrow", "condensed", "extralight", "extra light", "thin", "hairline", "cjk")
        def _allowed(f: FontInfo) -> bool:
            rp = Path(f.path).resolve()
            if project_fonts in rp.parents or rp.parent == script_dir.resolve():
                return True            # fonts the user placed in the project folder are always candidates
            nm = f"{f.family} {f.full_name} {rp.stem}".lower()
            return any(t in nm for t in family_allowlist) and not any(t in nm for t in deny)
        restricted = [f for f in fonts if _allowed(f)]
        if restricted:
            if verbose:
                print(f"Candidate pool restricted to {len(restricted)} Latin sans fonts "
                      f"(of {len(fonts)} installed); set FONT_FAMILY_ALLOWLIST = None to search all.")
            fonts = restricted
    default_path = str(default_font_path) if default_font_path else ""
    if default_path and Path(default_path).is_file() and not any(Path(f.path).resolve() == Path(default_path).resolve() for f in fonts):
        info = font_metadata(Path(default_path))
        if info.valid:
            fonts.insert(0, info)

    arr = np.asarray(image_rgb.convert("RGB"))
    configs: list[dict[str, Any]] = []
    if verbose:
        print("\n" + "=" * 78)
        print("FONT CALIBRATION: measuring original lettering before edits")
        print("=" * 78)
        print(f"Fonts discovered: {len(fonts)} usable, {len(invalid)} unreadable/invalid "
              f"(searched {len([d for d in dirs if Path(d).exists()])} directories)")

    for index, (box, raw_text) in enumerate(zip(boxes, original_texts), start=1):
        ix, iy, iw, ih = tm.inner_rect_of(box)
        crop = arr[iy:iy + ih, ix:ix + iw]
        an = analyses[index - 1] if analyses is not None else tm.analyze_interior(crop, cfg)
        text = " ".join((raw_text or "").split())
        notes: list[str] = list(an.notes)
        bold_hint = index in bold_box_indices

        base_cfg: dict[str, Any] = {
            "box_index": index,
            "original_text": text,
            "font_size_scale": 1.0,
            "text_color_rgb": list(an.fg_rgb),
            "background_rgb": list(an.bg_rgb),
            "original_ink_bbox_relative": list(an.ink_bbox_rel) if an.ink_bbox_rel else None,
            "original_ink_width_px": (an.ink_bbox_rel[2] - an.ink_bbox_rel[0]) if an.ink_bbox_rel else 0,
            "original_ink_height_px": (an.ink_bbox_rel[3] - an.ink_bbox_rel[1]) if an.ink_bbox_rel else 0,
            "alignment": tm.decide_alignment(an, iw),
            "contrast": round(an.contrast, 1),
            "tracking_em": 0.0,
            "word_spacing_em": 0.0,
            "alternatives": [],
            "ranking": [],
            "notes": notes,
        }

        rows: list[dict[str, Any]] = []
        if an.found and text:
            x0, y0, x1, y1 = an.ink_bbox_rel
            cov = (an.alpha * an.removal_mask)[y0:y1, x0:x1]
            rows = rank_fonts(text, cov, fonts, bold_hint)

        if rows:
            best = rows[0]
            size, stroke, fpath = int(best["size_px"]), int(best["stroke_width"]), best["path"]
            # --- stage 2: separate optimisation of spacing (font already chosen) ---
            tracking = 0.0
            n_gaps = len(text) - 1
            r0 = _render_features(text, fpath, size, stroke, 0.0)
            ow = an.ink_bbox_rel[2] - an.ink_bbox_rel[0]
            if r0 and n_gaps >= 2:
                tracking = float(np.clip((ow - r0[0]["w"]) / (n_gaps * size), *TRACKING_LIMITS))
                tracking = round(tracking, 4)
            rf = _render_features(text, fpath, size, stroke, tracking)
            width_resid = (rf[0]["w"] - ow) / ow if rf else None
            if abs(tracking) > TRACKING_SUSPECT_EM:
                notes.append(f"fitted letter spacing {tracking:+.3f}em is large: the OCR text may be wrong "
                             "or the true font is not installed - treat the font as a poor match")
            # --- baseline from the font's own descender extent for the original text ---
            below = tm.font_baseline_below(text, fpath, size, stroke)
            baseline_rel = an.ink_bbox_rel[3] - below
            conf, near = _confidence(rows)
            if near:
                notes.append("not separable from: " + ", ".join(near))
            alts = [{"font_path": r["path"], "font_size": int(r["size_px"]), "stroke_width": int(r["stroke_width"]),
                     "tracking_em": tracking, "score": round(r["score"], 4)}
                    for r in rows[1:4] if r["score"] - best["score"] < tm.MaskConfig().alt_font_score_margin]
            base_cfg.update({
                "font_path": fpath, "font_family": best["label"], "font_size": size,
                "stroke_width": stroke, "synthetic_bold": bool(best["synthetic_bold"]),
                "bold_target": bool(best["is_bold"] or best["synthetic_bold"]),
                "tracking_em": tracking, "baseline_rel": int(baseline_rel),
                "font_match_score": round(best["raw_score"], 5),
                "rendered_width_height_ratio": round(best["rendered_w"] / max(best["rendered_h"], 1), 4),
                "width_residual_after_tracking": None if width_resid is None else round(width_resid, 4),
                "calibration_confidence": conf, "alternatives": alts,
                "ranking": [{"font": r["label"], "path": r["path"], "score": round(r["score"], 4),
                             "errors": r["errors"], "size_px": r["size_px"]} for r in rows[:8]],
            })
        else:
            fallback = default_path if default_path else ""
            if not an.found:
                notes.append("no original ink found in the interior - font could not be measured")
            elif not text:
                notes.append("no OCR text - glyph shapes cannot be compared; default font used")
            else:
                notes.append("no installed font could render/rank this text; default font used")
            # NOT silent: confidence says fallback and the report/printout names the font
            size = max(8, int((an.ink_bbox_rel[3] - an.ink_bbox_rel[1]) / 0.72)) if an.ink_bbox_rel else max(8, int(ih * 0.5))
            base_cfg.update({
                "font_path": fallback, "font_family": Path(fallback).stem if fallback else "",
                "font_size": size, "stroke_width": 0, "synthetic_bold": False, "bold_target": bold_hint,
                "baseline_rel": an.baseline_rel_profile, "font_match_score": None,
                "rendered_width_height_ratio": None, "calibration_confidence": "fallback",
            })
        configs.append(base_cfg)
        if verbose:
            print(f"BOX #{index}: font={Path(base_cfg['font_path']).name or '(none)'}, size={base_cfg['font_size']}px, "
                  f"ink={base_cfg['original_ink_width_px']}x{base_cfg['original_ink_height_px']} px, "
                  f"tracking={base_cfg['tracking_em']:+.3f}em, confidence={base_cfg['calibration_confidence']}"
                  + (", bold" if base_cfg.get("bold_target") else ""))
            for n in notes:
                print(f"   note: {n}")

    out_path = Path(report_path) if report_path else script_dir / "font_calibration_report.json"
    try:
        out_path.write_text(json.dumps({
            "weights": RANKING_WEIGHTS, "fonts_usable": len(fonts), "fonts_invalid": len(invalid),
            "invalid_examples": [{"path": i.path, "error": i.error} for i in invalid[:10]],
            "boxes": configs}, indent=2, default=tm._json_default), encoding="utf-8")
        if verbose:
            print(f"Calibration report saved to: {out_path}")
    except OSError as exc:
        print(f"WARNING: could not write calibration report: {exc}")
    if verbose:
        print("Note: font identification is an estimate from pixels, not embedded metadata.")
        print("=" * 78 + "\n")
    return configs


if __name__ == "__main__":  # quick font inventory
    here = Path(__file__).resolve().parent
    good, bad = discover_fonts(default_font_dirs(here))
    print(f"{len(good)} usable fonts, {len(bad)} invalid")
    for f in good[:40]:
        print(f"  [{f.priority}] {f.label():40s} w{f.weight_class} wd{f.width_class} {f.path}")
