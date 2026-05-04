# Thali BEV

Automatically converts natural-angle thali tray images into a top-down Bird's Eye View (BEV), then runs food detection and Khana classification on the result. No manual annotation required.

This combines Task 3 (BEV normalisation) and Task 2 (food detection) of a Computer Vision project. Normalising the viewing angle first gives the detector a consistent top-down representation regardless of how the photo was taken.

## How it works

### Stage 1 — BEV transformation

1. Detect the 4 corners of the tray using three strategies tried in order:
   - **Otsu threshold** — fast; works when the tray is darker than the background
   - **GrabCut** — works when tray and background share a similar colour by fitting separate foreground/background colour models
   - **Canny edges** — fallback for high-contrast rims

2. Each candidate is validated — detections that span the whole image (corners at the image boundary) are rejected and the next strategy is tried.

3. The 4 corners are passed to `cv2.getPerspectiveTransform` to compute a homography matrix, then `cv2.warpPerspective` applies it.

4. Output dimensions are estimated from the detected corners so the tray's true aspect ratio is preserved.

### Stage 2 — Food detection

5. A fine-tuned **YOLO model** (`thali_detector.pt`) detects compartment regions in the BEV image.

6. Each detected crop is classified by a fine-tuned **ConvNeXt** model (`best_detection.pt`) trained to recognise 79 Khana classes (biryani, idli, gulab jamun, etc.).

7. Results are saved as annotated images and a JSON file for evaluation.

## Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Usage

Place images (`.jpg`, `.jpeg`, `.png`) in the `images/` directory, then run:

```bash
# Process all images — BEV warp + food detection (default)
uv run bev_pipeline.py

# Process a single image
uv run bev_pipeline.py --image foo.jpg

# BEV warp only, skip detection
uv run bev_pipeline.py --no-detect

# Save corner detection visualisation to output/debug/
uv run bev_pipeline.py --debug

# Set the base output size (default: 700px on the longer side)
uv run bev_pipeline.py --out-size 1000
```

### Detection options

```bash
# Adjust YOLO confidence threshold (default: 0.25)
uv run bev_pipeline.py --det-conf 0.4

# Use custom model paths
uv run bev_pipeline.py \
  --yolo models/thali_detector.pt \
  --classifier models/best_detection.pt \
  --classes data/classes.txt
```

## Output

| Path | Contents |
|------|----------|
| `output/bev/<name>_bev.jpg` | Perspective-corrected top-down view |
| `output/detected/<name>_det.jpg` | BEV image annotated with food labels and bounding boxes |
| `output/predictions.json` | All detections as JSON (label, confidence, bbox) |
| `output/debug/<name>_corners.jpg` | Original image with detected corners overlaid (`--debug` only) |

The pipeline prints a status line for each image:

```
[bev]   IMG_001.jpg → output/bev/IMG_001_bev.jpg  (847×560, ratio=0.81)
  biryani                        conf=0.909  bbox=[476, 124, 700, 384]
  garlic naan                    conf=0.908  bbox=[0, 295, 292, 507]
  gulab jamun                    conf=0.903  bbox=[510, 0, 700, 142]
[warn]  IMG_002.jpg — shallow angle (ratio=0.42), some stretching expected
[fail]  IMG_003.jpg — tray not detected (run with --debug to inspect)
```

The **foreshortening ratio** is `top_edge_length / bottom_edge_length`. A ratio of 1.0 means the image is already top-down; lower values indicate shallower angles and more perspective distortion in the output.

## Manual annotation fallback

If auto-detection fails for a specific image, `annotate.py` provides an interactive tool to click the 4 corners manually. Annotations are saved to `corners.json`.

```bash
uv run annotate.py
```

Controls: click 4 corners (TL → TR → BR → BL), then `s` to save, `r` to reset, `n` to skip.

To use the saved annotations in the pipeline, load `corners.json` and pass the corners directly to `warp_to_bev()`.

## Known limitations

**Extreme shallow angles (ratio < 0.3):** At very low viewing angles, the far edge of the tray is compressed to a handful of pixels. Perspective correction can geometrically undo the transform but cannot recover pixel detail that was never captured. The output will be stretched on the far side. This is a fundamental physics constraint — re-shooting at a steeper angle (ideally 45°+) is the correct fix.

**Same-colour tray on same-colour surface:** If the tray is indistinguishable in colour and texture from the surface it rests on, and the background provides no usable contrast, all three detection strategies may fail. Use `--debug` to inspect what each strategy found.

## Project structure

```
images/                  Input images
models/
  thali_detector.pt      Fine-tuned YOLO for thali compartment detection
  best_detection.pt      Fine-tuned ConvNeXt for Khana classification
data/
  classes.txt            79 Khana class names (one per line)
src/
  detector.py            YOLO + contour + colour-seg fallback detector
  classifier.py          ConvNeXt classifier wrapper
  pipeline.py            Chains detection → crop → classify
  preprocess.py          Mean-shift smoothing and CLAHE
  visualize.py           Draws bounding boxes and labels onto images
output/
  bev/                   Perspective-corrected top-down images
  detected/              Annotated images with food labels
  debug/                 Corner detection visualisations (--debug)
  predictions.json       All detections as JSON
bev_pipeline.py          Main pipeline — BEV warp then food detection
annotate.py              Manual corner annotation tool (fallback)
```
