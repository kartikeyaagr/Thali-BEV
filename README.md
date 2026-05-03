# Thali BEV

Automatically converts natural-angle thali tray images into a top-down Bird's Eye View (BEV) using perspective transformation. No manual annotation required.

This is Task 3 of a Computer Vision project. The BEV output normalises the viewing angle so a food detection model (Task 2) can run on a consistent top-down representation regardless of how the photo was taken.

## How it works

1. The pipeline detects the 4 corners of the tray in the input image using three strategies tried in order:
   - **Otsu threshold** — fast; works when the tray is darker than the background (e.g. black plastic tray on white table)
   - **GrabCut** — slower; works when tray and background share a similar colour (e.g. beige tray on white table) by fitting separate colour models for foreground and background
   - **Canny edges** — fallback for high-contrast rims

2. Each candidate detection is validated — detections that span the whole image (corners at the image boundary) are rejected and the next strategy is tried.

3. The 4 corners are passed to `cv2.getPerspectiveTransform` to compute a homography matrix, then `cv2.warpPerspective` applies it.

4. The output dimensions are estimated from the detected corners so the aspect ratio of the tray is preserved — a 4:3 tray is not squished into a 1:1 square.

## Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Usage

Place images (`.jpg`, `.jpeg`, `.png`) in the `images/` directory, then run:

```bash
# Process all images
uv run bev_pipeline.py

# Process a single image
uv run bev_pipeline.py --image foo.jpg

# Save corner detection visualisation to output/debug/
uv run bev_pipeline.py --debug

# Set the base output size (default: 700px on the longer side)
uv run bev_pipeline.py --out-size 1000
```

Output is written to `output/bev/`.

## Output

| Path | Contents |
|------|----------|
| `output/bev/<name>_bev.jpg` | Perspective-corrected top-down view |
| `output/debug/<name>_corners.jpg` | Original image with detected corners overlaid (`--debug` only) |

The pipeline prints a status line for each image:

```
[done]  IMG_001.jpg → output/bev/IMG_001_bev.jpg  (847×560, ratio=0.81)
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
images/          Input images
output/
  bev/           Perspective-corrected outputs
  debug/         Corner detection visualisations (--debug)
bev_pipeline.py  Main pipeline — auto-detects corners and warps
annotate.py      Manual corner annotation tool (fallback)
```
