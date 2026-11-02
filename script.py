import os
import glob
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def get_local_font():
    """Finds the first .ttf or .otf font file in the script directory."""
    font_files = glob.glob("*.ttf") + glob.glob("*.otf")
    if font_files:
        print(f"Using local font file: {font_files[0]}")
        return font_files[0]
    print("Warning: No local font file (.ttf/.otf) found. Falling back to default.")
    return None

def find_red_boxes(image):
    """Detects red box contours using HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color range in HSV space
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 15:  # Filter out tiny noise
            boxes.append((x, y, w, h))

    # Sort boxes top-to-bottom, then left-to-right
    boxes.sort(key=lambda b: (b[1] // 30, b[0]))
    return boxes

def fit_font_size(font_path, text, target_w, target_h):
    """Iteratively finds the precise font point size matching target bounds."""
    if not font_path:
        return ImageFont.load_default(), 10

    best_font = ImageFont.truetype(font_path, 10)

    # Step through sizes to fit target dimensions
    for sz in range(8, 200):
        test_font = ImageFont.truetype(font_path, sz)
        bbox = test_font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w <= target_w and h <= target_h:
            best_font = test_font
        else:
            break

    return best_font

def process_image(image_path="original.png", output_path="output.png"):
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found in directory.")
        return

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    local_font_path = get_local_font()

    boxes = find_red_boxes(img)
    print(f"\nDetected {len(boxes)} red box(es).")

    replacements = []

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        pad = 4
        crop_bgr = img[y+pad:y+h-pad, x+pad:x+w-pad]

        # Read original value using OCR
        ocr_results = reader.readtext(crop_bgr)
        
        if ocr_results:
            bbox, orig_text, _ = ocr_results[0]
            pts = np.array(bbox, dtype=np.int32)
            tx_min, ty_min = np.min(pts, axis=0)
            tx_max, ty_max = np.max(pts, axis=0)
            orig_text_h = max(1, ty_max - ty_min)
        else:
            orig_text = "?"
            orig_text_h = int(h * 0.6)

        # Highlight box preview
        preview = img.copy()
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)

        window_name = f"Box {idx} of {len(boxes)}"
        cv2.imshow(window_name, preview)
        cv2.waitKey(100)

        print("=" * 50)
        print(f" BOX [{idx}/{len(boxes)}]")
        print(f" ORIGINAL VALUE : '{orig_text}'")
        print("=" * 50)

        user_input = input(f"Enter NEW value for Box #{idx} (Original: '{orig_text}'): ")
        cv2.destroyWindow(window_name)

        if user_input.strip() != "":
            replacements.append({
                "box": (x, y, w, h),
                "orig_h": orig_text_h,
                "new_text": user_input.strip()
            })

    cv2.destroyAllWindows()

    # White font color forced
    WHITE_COLOR = (255, 255, 255)

    # Render replacement text inside boxes
    for item in replacements:
        x, y, w, h = item["box"]
        new_text = item["new_text"]
        orig_h = item["orig_h"]

        # Erase inner area of red box (samples inner background or fills transparent/dark)
        pad = 3
        
        # Target constraints for replacement string
        target_h = int(h * 0.70)      # Max height is 70% of inner box height
        target_w = int(w * 0.85)      # Max width is 85% of inner box width
        
        font = fit_font_size(local_font_path, new_text, target_w, target_h)

        # Measure rendered text size for exact center positioning
        text_bbox = font.getbbox(new_text)
        rendered_w = text_bbox[2] - text_bbox[0]
        rendered_h = text_bbox[3] - text_bbox[1]

        center_x = x + (w // 2)
        center_y = y + (h // 2)

        text_x = center_x - (rendered_w // 2) - text_bbox[0]
        text_y = center_y - (rendered_h // 2) - text_bbox[1]

        # Erase old characters by clearing the original position region
        bg_sample = img_rgb[y+pad:y+h-pad, x+pad:x+w-pad]
        bg_color = tuple(np.median(bg_sample, axis=(0, 1)).astype(int))
        draw.rectangle([x + pad, y + pad, x + w - pad, y + h - pad], fill=bg_color)

        # Draw replacement text in pure white
        draw.text((text_x, text_y), new_text, fill=WHITE_COLOR, font=font)

    pil_img.save(output_path)
    print(f"\nCompleted successfully! Image saved as: {output_path}")

if __name__ == "__main__":
    process_image()