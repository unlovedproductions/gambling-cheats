#!/usr/bin/env python3
"""Download open-licensed sans-serif fonts when the right one is not installed.

Used by text_replacer_complete.py after font calibration when the best installed
font is a poor / ambiguous match.  Fonts come ONLY from the public ``google/fonts``
repository on GitHub (SIL OFL / Apache 2.0 licensed), listed at run time through
the GitHub contents API so file names never go stale.

Important limits (the caller reports these to the user):
* Proprietary faces (Arial, Helvetica, Segoe UI, SF Pro, Calibri ...) cannot be
  downloaded legally.  Metric/outline-compatible open substitutes ARE offered
  (Arimo = Liberation Sans = Arial-compatible, Carlito = Calibri-compatible).
  They are always labelled as substitutes - never presented as the original.
* A downloaded font is only a better CANDIDATE.  Calibration re-ranks it against
  the pixels; it is kept for a box only if it actually scores better.

Standalone use:
    python font_fetcher.py --list
    python font_fetcher.py "Roboto" "Lato"          # download into ./fonts
    python font_fetcher.py --core                   # the core set
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

API_ROOT = "https://api.github.com/repos/google/fonts/contents/"
ALLOWED_HOSTS = ("api.github.com", "raw.githubusercontent.com")
MAX_FONT_BYTES = 6 * 1024 * 1024      # matches font_calibration.MAX_FONT_BYTES (larger files are ignored there)
MAX_FILES_PER_FAMILY = 6
WANTED_STYLES = ("regular", "medium", "semibold", "bold")
USER_AGENT = "text-replacer-font-fetcher/1.0"


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    repo_dir: str            # directory inside google/fonts
    core: bool = False       # offered by default
    note: str = ""
    equivalent_to: tuple[str, ...] = ()   # installed families that make this one redundant
    alt_dirs: tuple[str, ...] = ()        # other google/fonts directories to try if repo_dir is 404


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry("Arimo", "ofl/arimo", True, "open substitute for Arial / Helvetica / Liberation Sans",
                 ("liberation sans",)),
    CatalogEntry("Roboto", "ofl/roboto", True, alt_dirs=("apache/roboto",)),
    CatalogEntry("Open Sans", "ofl/opensans", True),
    CatalogEntry("Inter", "ofl/inter", True),
    CatalogEntry("Lato", "ofl/lato", True),
    CatalogEntry("Noto Sans", "ofl/notosans", True),
    CatalogEntry("Source Sans 3", "ofl/sourcesans3", True),
    CatalogEntry("Carlito", "ofl/carlito", False, "open substitute for Calibri"),
    CatalogEntry("IBM Plex Sans", "ofl/ibmplexsans"),
    CatalogEntry("Work Sans", "ofl/worksans"),
    CatalogEntry("Public Sans", "ofl/publicsans"),
    CatalogEntry("Fira Sans", "ofl/firasans"),
    CatalogEntry("Montserrat", "ofl/montserrat"),
    CatalogEntry("Poppins", "ofl/poppins"),
    CatalogEntry("Nunito Sans", "ofl/nunitosans"),
    CatalogEntry("DM Sans", "ofl/dmsans"),
    CatalogEntry("Rubik", "ofl/rubik"),
    CatalogEntry("Barlow", "ofl/barlow"),
)

PROPRIETARY_NOTE = (
    "Arial, Helvetica, Segoe UI, SF Pro, Calibri and similar faces are proprietary and are not downloaded. "
    "If the original is one of them, install it from your own system (copy the .ttf into ./fonts) - "
    "the open substitutes are close in shape but not identical."
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def missing_entries(installed_families: list[str], scope: str = "core",
                    extra: Optional[list[CatalogEntry]] = None) -> list[CatalogEntry]:
    """Catalog families that are not installed.  ``scope``: 'core' or 'all'."""
    have = {_norm(f) for f in installed_families if f}
    out: list[CatalogEntry] = []
    for e in list(CATALOG) + list(extra or []):
        if scope == "core" and not e.core:
            continue
        if _norm(e.family) in have or any(_norm(a) in have for a in e.equivalent_to):
            continue
        out.append(e)
    return out


def find_entry(name: str) -> Optional[CatalogEntry]:
    n = _norm(name)
    for e in CATALOG:
        if _norm(e.family) == n:
            return e
    return None


def _check_host(url: str) -> None:
    host = urllib.parse.urlparse(url).hostname or ""
    if urllib.parse.urlparse(url).scheme != "https" or host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing to fetch from {host or url!r}")


def _http_get(url: str, timeout: float, limit: Optional[int] = None) -> bytes:
    _check_host(url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - host allow-listed above
        _check_host(resp.geturl())            # a redirect may not leave the allow-list
        data = resp.read(limit + 1 if limit else -1)
    if limit and len(data) > limit:
        raise ValueError(f"download larger than {limit} bytes")
    return data


def _pick_files(listing: list[dict]) -> list[dict]:
    """Upright regular/medium/semibold/bold files (static preferred, else the upright variable file)."""
    ttfs = [f for f in listing if f.get("type") == "file" and f.get("name", "").lower().endswith(".ttf")
            and "italic" not in f["name"].lower() and f.get("download_url")]
    static = [f for f in ttfs if "[" not in f["name"]
              and any(s in f["name"].lower() for s in WANTED_STYLES)
              and not re.search(r"(light|thin|black|heavy|extra|condensed|narrow)", f["name"].lower())]
    chosen = static or [f for f in ttfs if "[" in f["name"]]
    chosen.sort(key=lambda f: ("regular" not in f["name"].lower(), f["name"].lower()))
    return chosen[:MAX_FILES_PER_FAMILY]


def _validate_font(path: Path) -> None:
    from fontTools.ttLib import TTFont
    from PIL import ImageFont
    f = TTFont(str(path), lazy=True, fontNumber=0)
    try:
        if not f.getBestCmap():
            raise ValueError("font has no character map")
    finally:
        f.close()
    ImageFont.truetype(str(path), size=24)     # must also load in Pillow


def download_family(entry: CatalogEntry, dest_dir: Path, timeout: float = 30.0,
                    log: Callable[[str], None] = print) -> tuple[list[Path], list[str]]:
    """Download one family into ``dest_dir``.  Returns (saved paths, error messages). Never raises."""
    saved: list[Path] = []
    errors: list[str] = []
    dest_dir = Path(dest_dir)
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return saved, [f"{entry.family}: cannot create {dest_dir} ({exc})"]
    listing = None
    last_err = ""
    for repo_dir in (entry.repo_dir, *entry.alt_dirs):
        try:
            listing = json.loads(_http_get(API_ROOT + urllib.parse.quote(repo_dir), timeout).decode("utf-8"))
            if not isinstance(listing, list):
                raise ValueError(str(listing)[:200])
            break
        except urllib.error.HTTPError as exc:
            listing = None
            last_err = f"{entry.family}: listing failed (HTTP {exc.code}) for {repo_dir}"
            if exc.code in (403, 429):
                last_err += " - GitHub rate limit; try again in an hour or copy the .ttf into ./fonts yourself"
                break
        except Exception as exc:
            listing = None
            last_err = f"{entry.family}: listing failed ({type(exc).__name__}: {exc})"
    if listing is None:
        return saved, [last_err]

    files = _pick_files(listing)
    if not files:
        return saved, [f"{entry.family}: no suitable .ttf files found in {entry.repo_dir}"]
    for f in files:
        name = re.sub(r"[^\w.\-]", "_", f["name"])
        target = dest_dir / name
        if target.is_file():
            saved.append(target)
            continue
        tmp = target.with_suffix(".part")
        try:
            data = _http_get(f["download_url"], timeout, MAX_FONT_BYTES)
            tmp.write_bytes(data)
            _validate_font(tmp)
            tmp.replace(target)
            saved.append(target)
            log(f"   downloaded {name} ({len(data) // 1024} KB)")
        except Exception as exc:
            errors.append(f"{entry.family}/{f['name']}: {type(exc).__name__}: {exc}")
            try:
                tmp.unlink()
            except OSError:
                pass
    return saved, errors


def download_families(entries: list[CatalogEntry], dest_dir: Path, timeout: float = 30.0,
                      log: Callable[[str], None] = print) -> tuple[list[Path], list[str]]:
    all_saved: list[Path] = []
    all_err: list[str] = []
    for e in entries:
        log(f" - {e.family}" + (f" ({e.note})" if e.note else ""))
        s, err = download_family(e, dest_dir, timeout, log)
        all_saved += s
        all_err += err
    return all_saved, all_err


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("families", nargs="*", help="catalog family names, e.g. Roboto Lato")
    ap.add_argument("--list", action="store_true", help="show the catalog")
    ap.add_argument("--core", action="store_true", help="download the core set")
    ap.add_argument("--all", action="store_true", help="download every catalog family")
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent / "fonts"))
    args = ap.parse_args(argv)
    if args.list:
        for e in CATALOG:
            print(f"{'*' if e.core else ' '} {e.family:16s} {e.repo_dir:22s} {e.note}")
        print("\n* = core set\n" + PROPRIETARY_NOTE)
        return 0
    chosen: list[CatalogEntry] = []
    if args.all:
        chosen = list(CATALOG)
    elif args.core:
        chosen = [e for e in CATALOG if e.core]
    for name in args.families:
        e = find_entry(name)
        if e is None:
            print(f"unknown family {name!r} (see --list)")
            return 2
        chosen.append(e)
    if not chosen:
        ap.print_help()
        return 2
    saved, errs = download_families(chosen, Path(args.dest))
    print(f"\n{len(saved)} file(s) in {args.dest}")
    for e in errs:
        print("ERROR:", e)
    return 1 if errs and not saved else 0


if __name__ == "__main__":
    sys.exit(main())
