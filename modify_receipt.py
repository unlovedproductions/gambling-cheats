#!/usr/bin/env python3
# Six typed text blocks. Collects input for each box and renders a labeled
# preview. Does not load or modify a receipt screenshot.

from datetime import datetime
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont
import pytz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "blocks_preview.png")
PST = pytz.timezone("US/Pacific")

AVATAR_RE = re.compile(r"^[A-Za-z]$")
NAME_RE = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+)*$")
HANDLE_RE = re.compile(r"^\$[A-Za-z]+(?:-[A-Za-z]+)*$")
AMOUNT_RE = re.compile(r"^-?\$\d+(?:\.\d{1,2})?$")
MONTH_DAY_TIME_RE = re.compile(
    r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"(?:[12]\d|3[01]|0?[1-9]) at (?:1[0-2]|0?[1-9]):[0-5]\d (?:AM|PM)$"
)
TIME_12HR_RE = re.compile(r"^(?:1[0-2]|0?[1-9]):[0-5]\d (?:AM|PM)$")


def get_font(font_name_base, default_size):
    common_paths = [
        os.path.join(SCRIPT_DIR, f"{font_name_base}.ttf"),
        f"{font_name_base}.ttf",
        f"{font_name_base}-Medium.ttf",
        f"{font_name_base}-Bold.ttf",
        f"{font_name_base}-Regular.ttf",
        f"/usr/share/fonts/truetype/roboto/{font_name_base}.ttf",
        f"/System/Library/Fonts/{font_name_base}.ttf",
        f"C:/Windows/Fonts/{font_name_base}.ttf",
        "arial.ttf",
        "sans-serif.ttf",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, default_size)
    print(
        f"Warning: Could not find '{font_name_base}.ttf'. Using default PIL font."
    )
    return ImageFont.load_default()


def now_pst():
    return datetime.now(PST)


def default_month_day_time():
    return now_pst().strftime("%b %-d at %-I:%M %p")


def default_time_12hr():
    return now_pst().strftime("%-I:%M %p")


def prompt_block(label, text_type, example, pattern, default):
    hint = f"{label} [{text_type}] (e.g. '{example}', Enter for '{default}'): "
    while True:
        value = input(hint).strip() or default
        if pattern.fullmatch(value):
            return value
        print(f"Invalid. Expected {text_type}, like '{example}'.")


def get_user_input():
    print("=== Text block inputs (not a receipt editor) ===")
    return {
        "avatar": prompt_block(
            "1 avatar initial",
            "letters only",
            "V",
            AVATAR_RE,
            "V",
        ),
        "name": prompt_block(
            "2 display name",
            "letters only",
            "Vikki Valdez",
            NAME_RE,
            "Vikki Valdez",
        ),
        "handle": prompt_block(
            "3 handle",
            "special chars and letters",
            "$Vikki-Valdez",
            HANDLE_RE,
            "$Vikki-Valdez",
        ),
        "amount": prompt_block(
            "4 amount",
            "special chars and numbers",
            "-$5",
            AMOUNT_RE,
            "-$5",
        ),
        "month_day_time": prompt_block(
            "5 month day time",
            "month day and 12hr time",
            "Jul 28 at 3:45 AM",
            MONTH_DAY_TIME_RE,
            default_month_day_time(),
        ),
        "clock": prompt_block(
            "6 status clock",
            "12hr time only",
            "3:45 AM",
            TIME_12HR_RE,
            default_time_12hr(),
        ),
    }


def render_blocks(blocks):
    img = Image.new("RGB", (920, 560), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    label_font = get_font("Roboto-Regular", 16)
    type_font = get_font("Roboto-Regular", 14)

    rows = [
        ("1", "letters only", blocks["avatar"], get_font("Roboto-Medium", 22)),
        ("2", "letters only", blocks["name"], get_font("Roboto-Medium", 22)),
        (
            "3",
            "special chars and letters",
            blocks["handle"],
            get_font("Roboto-Regular", 20),
        ),
        (
            "4",
            "special chars and numbers",
            blocks["amount"],
            get_font("Roboto-Bold", 56),
        ),
        (
            "5",
            "month day and 12hr time",
            blocks["month_day_time"],
            get_font("Roboto-Regular", 16),
        ),
        (
            "6",
            "12hr time only",
            blocks["clock"],
            get_font("Roboto-Regular", 14),
        ),
    ]

    y = 28
    for number, text_type, text, font in rows:
        draw.text((24, y), number, fill=(120, 120, 120), font=label_font)
        draw.text((56, y), text, fill=(0, 0, 0), font=font)
        bbox = draw.textbbox((56, y), text, font=font)
        type_y = bbox[3] + 4
        draw.text((56, type_y), text_type, fill=(140, 140, 140), font=type_font)
        type_bbox = draw.textbbox((56, type_y), text_type, font=type_font)
        y = type_bbox[3] + 20

    img.save(OUTPUT_PATH)
    return OUTPUT_PATH


def main():
    blocks = get_user_input()
    for key, value in blocks.items():
        print(f"{key}: {value}")
    path = render_blocks(blocks)
    print(f"Labeled preview saved to {path}")
    return True


if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)
