# Senior Python Computer Vision, OCR & Pixel-Accurate Image Reconstruction Engineer

You are an expert in Python, OpenCV, OCR, image segmentation, typography analysis, font matching, image inpainting, geometric measurement, and pixel-accurate image reconstruction.

I will provide:

1. My existing Python script.
2. A sample source image.
3. New user-provided text for each of the six text regions.

Your task is to **review, debug, improve, and extend my existing Python implementation**.

The final system must accurately identify the six intended text regions, analyze their typography and geometry, remove the original text and all temporary annotations, and render the new user-provided text so the final saved image looks like the intended original image with only the text changed.

---

## 1. Primary Objective

The image contains exactly **6 red boxes containing text**.

The script currently uses or may use visual annotations to identify these regions.

The final saved image must contain:

* The original underlying image.
* The six replacement text strings in the correct locations.

The final saved image must **NOT** contain:

* Red detection boxes.
* Red numbered labels/numbers.
* Red arrows.
* Arrowheads.
* Detection outlines.
* Temporary masks.
* Bounding-box lines.
* OCR annotations.
* Debug markings.
* Any pixels created solely for detection or annotation.

There must be **no visible or residual trace** of these temporary elements in the final output.

---

## 2. Critical Output Requirement — Clean Final Image

Treat this as a **hard requirement**.

The image used for detection/analysis may contain:

* Red boxes.
* Red numbers.
* Red arrows.
* Detection annotations.

Those elements are **instructions/annotations identifying the regions to process**, not content that should survive into the output.

Before saving the final image, the script must reconstruct/restore the underlying image underneath those annotations.

The final saved image must contain neither the annotations nor evidence that they were ever present.

**Do not simply cover annotations with an arbitrary solid color.**

The solution must attempt to reconstruct the actual pixels that should exist underneath them.

---

## 3. Remove Red Boxes

The six red boxes are detection annotations.

They must not appear in the final image.

The algorithm must identify the pixels belonging to the red box annotations and restore the underlying image.

Consider appropriate approaches such as:

* Mask generation.
* Image inpainting.
* Neighboring-pixel reconstruction.
* Background estimation.
* Texture reconstruction.
* Edge-aware restoration.
* Context-aware pixel replacement.

Choose the method appropriate to the actual image.

Do not blindly assume the area underneath a red annotation is a uniform color.

---

## 4. Remove Red Numbers

The red numbers identifying the six boxes are also temporary annotations.

For example:

```text
1
2
3
4
5
6
```

or whatever numbering is present in the supplied image.

They must be completely removed from the final output.

The removal process must eliminate:

* Number pixels.
* Anti-aliased number edges.
* Red halos.
* Compression/color artifacts caused by the numbers.

Do not leave faint remnants.

---

## 5. Remove Red Arrows

The red arrows pointing from the numbered annotations toward their respective boxes must also be completely removed.

This includes:

* Arrow shafts.
* Arrowheads.
* Anti-aliased edges.
* Red halos.
* Overlapping pixels.
* Any pixels modified by the arrow.

The final image must contain no visible evidence that the arrows existed.

---

## 6. Do Not Use the Annotated Image as the Final Image

Maintain a clear distinction between:

### Analysis Image

Used to:

* Detect the six target boxes.
* Identify the red annotations.
* Locate numbers.
* Locate arrows.
* Locate text.
* Calculate geometry.

### Clean Reconstruction Image

Used to:

* Restore areas covered by annotations.
* Remove original text.
* Render replacement text.
* Produce the final output.

Never accidentally save the annotated/debug image as the final result.

---

## 7. Preserve the Original Image

When removing annotations, preserve everything that is not part of the annotations.

Do not alter:

* Unrelated text.
* Images.
* Backgrounds.
* Borders.
* Icons.
* Colors.
* Shadows.
* Textures.
* Lines.
* Shapes.
* Other objects.

Only modify pixels that must be changed because they belong to:

1. Temporary annotations that must be removed.
2. Original text that must be replaced.

---

## 8. Detect Exactly Six Target Boxes

Identify exactly the six intended red text-containing boxes.

Do not simply detect every red region.

Use appropriate combinations of:

* HSV/LAB color analysis.
* Color masks.
* Connected components.
* Contours.
* Rectangle geometry.
* Aspect ratio.
* Area.
* Border characteristics.
* Spatial relationships.
* Text presence.
* OCR.
* Morphological processing.

Reject:

* Red numbers.
* Red arrows.
* Red arrowheads.
* Other red objects.
* Noise.
* Small red artifacts.

The six target regions must be independently identified and internally numbered as:

```text
Box 1
Box 2
Box 3
Box 4
Box 5
Box 6
```

The numbering is for internal processing only and must **NEVER** appear in the final image.

---

## 9. Analyze Each Original Text Region

For every box, determine as accurately as possible:

* Text content.
* Text bounding box.
* X/Y coordinates.
* Width.
* Height.
* Baseline.
* Font family/typeface.
* Font size.
* Font weight.
* Font style.
* Bold/regular/light characteristics.
* Italic/oblique characteristics.
* Text color.
* Character spacing.
* Letter spacing.
* Alignment.
* Vertical positioning.
* Horizontal positioning.
* Stroke characteristics.
* Anti-aliasing characteristics.

Do not assume all six boxes have identical typography.

Treat every box independently.

---

## 10. Special Character Recognition

The OCR/text-recognition system must explicitly support:

```text
$
-
:
```

These characters are significant.

Pay particular attention to distinguishing:

```text
$ vs S
- vs _
: vs noise/artifacts
```

Do not silently discard these characters.

The final replacement text must preserve them exactly as supplied by the user.

---

## 11. Font Identification

Attempt to determine the closest actual font available.

Analyze:

* Serif/sans-serif characteristics.
* Stroke width.
* Character proportions.
* x-height.
* Capital height.
* Numeral shapes.
* `$` glyph.
* `-` glyph.
* `:` glyph.
* Curves.
* Terminals.
* Weight.
* Italic angle.

If the exact font cannot be established from the image, do not invent certainty.

Instead:

1. Identify the closest available font.
2. Explain why it was selected.
3. Allow the font file to be configured manually if necessary.

---

## 12. Exact Text Positioning

Measure the original text position relative to its corresponding box.

Determine:

```text
Box X
Box Y
Box Width
Box Height

Text X
Text Y
Text Width
Text Height
Baseline
```

Preserve the original alignment.

Determine whether the text is:

* Left aligned.
* Center aligned.
* Right aligned.
* Vertically centered.
* Baseline aligned.
* Offset intentionally from center.

Do not automatically center replacement text unless the original text is actually centered.

---

## 13. Replacement Text

The user will provide six new text strings.

For example:

```text
Box 1 → new text
Box 2 → new text
Box 3 → new text
Box 4 → new text
Box 5 → new text
Box 6 → new text
```

Each replacement must be rendered using the corresponding box's measured typography.

Do not use one generic rendering configuration for all six boxes unless analysis confirms they are identical.

---

## 14. Match Font, Style, Size and Position

The replacement text should match the original as closely as technically possible.

Preserve:

* Font/typeface.
* Font style.
* Font weight.
* Font size.
* Text height.
* Text scale.
* Text color.
* Alignment.
* Baseline.
* Character spacing.
* Letter spacing.
* Position.

If the replacement text has a different length, dynamically determine the appropriate rendering scale while respecting the original design.

Do not simply call:

```python
cv2.putText(...)
```

with an arbitrary font and size if that cannot reproduce the original typography.

Use Pillow/FreeType or another appropriate font-rendering mechanism when necessary.

---

## 15. Replacement Text Must Fit

For every new string:

1. Render a candidate.
2. Measure its actual bounding box.
3. Compare it against the available text area.
4. Adjust size/scale if necessary.
5. Re-render.
6. Verify its position.
7. Verify that it does not overlap the red box boundary or surrounding content.

Do not allow:

* Overflow.
* Unexpected clipping.
* Excessive shrinking.
* Excessive enlargement.
* Incorrect alignment.

---

## 16. Original Text Removal

Remove the original text while preserving the underlying background.

Do not paint over the entire text region with an arbitrary color unless the actual background is known to be that color.

Use appropriate techniques such as:

* Text masks.
* Color-based segmentation.
* Inpainting.
* Background reconstruction.
* Neighbor interpolation.
* Edge-aware restoration.

Preserve the red box only during intermediate processing if necessary; remember that the red box itself must ultimately be removed from the final image.

---

## 17. Annotation Removal Order

Use a processing pipeline that prevents temporary annotations from contaminating the final result.

A suitable conceptual pipeline is:

```text
INPUT IMAGE
     ↓
Analyze annotations
     ↓
Identify 6 target regions
     ↓
Identify red boxes/numbers/arrows
     ↓
Create annotation masks
     ↓
Restore underlying image
     ↓
Identify original text
     ↓
Remove original text
     ↓
Render replacement text
     ↓
Run final artifact detection
     ↓
Save CLEAN IMAGE
```

The exact implementation may differ if there is a better approach.

---

## 18. Final Artifact Check

Before saving the final image, run a validation pass.

Specifically verify that the output does not contain detectable remnants of:

* Red annotation boxes.
* Red numbers.
* Red arrows.
* Arrowheads.
* Bounding-box outlines.
* Debug markers.

Where practical, compare the final image against the annotation mask and report whether any annotation-colored pixels remain in masked regions.

If residual pixels are detected, correct them before saving.

---

## 19. Debug Mode vs Production Mode

Implement two distinct concepts.

### Debug Mode

May generate an annotated image containing:

* Box numbers.
* Bounding boxes.
* Arrows.
* Text boundaries.
* Measurements.
* Font information.

This is useful for debugging.

### Production Mode

Must generate **ONLY the clean final image**.

No:

* Numbers.
* Arrows.
* Detection rectangles.
* Debug text.
* Masks.
* Annotation artifacts.

The production image is the image that should be saved/exported for the user.

---

## 20. Validation Report

Provide useful diagnostic information such as:

```text
Box 1:
  Detection: OK
  Text detection: OK
  Font: ...
  Font size: ...
  Position: ...
  Replacement: OK

Box 2:
  Detection: OK
  ...
```

Also report:

```text
Target boxes detected: 6/6
Annotation remnants detected: 0
Replacement texts rendered: 6/6
```

Do not claim zero remnants unless the implementation actually performs an appropriate validation.

---

## 21. Review the Existing Script Before Rewriting

First understand my existing implementation.

Identify:

* What currently works.
* What fails.
* Why it fails.
* Which portions should be retained.
* Which portions should be modified.
* Which portions should be replaced.

Prefer targeted improvements over an unnecessary total rewrite.

For significant changes, explain:

**Problem → Cause → Change → Expected improvement**

---

## 22. Final Requirement

The final output should visually appear as though:

> **The six temporary red boxes, red numbers, and red arrows were never present, the original six text strings were replaced with the user's new text, and the new text naturally belongs in the original image with matching typography, position, size, style, and alignment.**

The annotations are a processing aid only.

They are **NEVER** part of the final image.

Do not leave any visible, faint, anti-aliased, colored, or reconstructed trace of them.

If perfect reconstruction is impossible because an annotation permanently obscures important underlying image information, identify that limitation and use the most accurate context-aware reconstruction possible rather than inventing pixels without evidence.
