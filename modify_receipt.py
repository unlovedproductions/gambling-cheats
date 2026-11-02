import os
import glob
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont

# Initialize EasyOCR with symbol detection enabled
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

    # Red color range in HSV space (covers wraparound 0-10 and 170-180)
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

def read_box_text(crop_bgr):
    """Attempts OCR reading allowing special characters, with graceful fallback."""
    # Allow numbers, uppercase, lowercase, and common special characters
    allow_list = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_+=[]{}|;:,.<>?/'
    
    ocr_results = reader.readtext(
        crop_bgr, 
        allowlist=allow_list,
        paragraph=False
    )
    
    if ocr_results:
        # Get recognized text
        text = ocr_results[0][1].strip()
        return text if text else "[Unrecognized Characters]"
    return "[Unrecognized Characters]"

def fit_font_size(font_path, text, target_w, target_h):
    """Iteratively calculates font point size matching target bounds."""
    if not font_path:
        return ImageFont.load_default()

    best_font = ImageFont.truetype(font_path, 10)

    # Step through sizes to find maximum fit within box constraints
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

        # Extract text using expanded character whitelist
        orig_text = read_box_text(crop_bgr)

        # Highlight box in OpenCV window
        preview = img.copy()
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)

        window_name = f"Box {idx} of {len(boxes)}"
        cv2.imshow(window_name, preview)
        cv2.waitKey(100)

        print("=" * 50)
        print(f" BOX [{idx}/{len(boxes)}]")
        print(f" ORIGINAL DETECTED VALUE : '{orig_text}'")
        print(f" Box Position           : (x: {x}, y: {y}) | Size: {w}x{h} px")
        print("=" * 50)

        user_input = input(f"Enter NEW value for Box #{idx}: ")
        cv2.destroyWindow(window_name)

        if user_input.strip() != "":
            replacements.append({
                "box": (x, y, w, h),
                "new_text": user_input.strip()
            })

    cv2.destroyAllWindows()

    WHITE_COLOR = (255, 255, 255)

    # Render replacement text inside detected red boxes
    for item in replacements:
        x, y, w, h = item["box"]
        new_text = item["new_text"]

        pad = 3
        
        # Define printable interior target bounds
        target_h = int(h * 0.70)
        target_w = int(w * 0.85)
        
        font = fit_font_size(local_font_path, new_text, target_w, target_h)

        # Measure rendered string for exact center alignment
        text_bbox = font.getbbox(new_text)
        rendered_w = text_bbox[2] - text_bbox[0]
        rendered_h = text_bbox[3] - text_bbox[1]

        center_x = x + (w // 2)
        center_y = y + (h // 2)

        text_x = center_x - (rendered_w // 2) - text_bbox[0]
        text_y = center_y - (rendered_h // 2) - text_bbox[1]

        # Erase old interior by sampling background color inside box
        bg_sample = img_rgb[y+pad:y+h-pad, x+pad:x+w-pad]
        bg_color = tuple(np.median(bg_sample, axis=(0, 1)).astype(int))
        draw.rectangle([x + pad, y + pad, x + w - pad, y + h - pad], fill=bg_color)

        # Draw replacement text in pure white
        draw.text((text_x, text_y), new_text, fill=WHITE_COLOR, font=font)

    pil_img.save(output_path)
    print(f"\nProcessing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    process_image()