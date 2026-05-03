"""
Warp thali images to a top-down Bird's Eye View automatically.

Detects the tray boundary — no manual annotation needed.

Output: output/bev/<name>_bev.jpg
        output/debug/<name>_corners.jpg   (with --debug)

Usage:
  uv run bev_pipeline.py                   # all images in images/
  uv run bev_pipeline.py --image foo.jpg   # single image
  uv run bev_pipeline.py --debug           # save corner visualisation
"""

import argparse
import os
import cv2
import numpy as np
import glob


IMAGE_DIR = "images"
OUTPUT_BEV = "output/bev"
OUTPUT_DEBUG = "output/debug"
DETECT_W = 800  # downsample to this width for detection; scales back at the end


def order_points(pts):
    """Order 4 points as [TL, TR, BR, BL]."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    out = np.zeros((4, 2), dtype="float32")
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(diff)]
    out[3] = pts[np.argmax(diff)]
    return out


def _is_valid_quad(corners, img_shape):
    """
    Reject detections that are clearly wrong:
    - Quad covers > 90% of the image → whole-image detected, not the tray.
    - Any corner is within 1% of the image boundary → same problem.
    """
    h, w = img_shape[:2]
    pts = np.array(corners, dtype=np.float32)
    quad_area = cv2.contourArea(pts)
    if quad_area > 0.90 * h * w:
        return False
    for x, y in corners:
        if x < 0.01 * w or x > 0.99 * w or y < 0.01 * h or y > 0.99 * h:
            return False
    return True


def _quad_from_mask(binary, img_area, img_shape, min_frac=0.08):
    """
    Find the largest blob in a binary mask and fit a quadrilateral to it.
    Returns ordered [TL, TR, BR, BL] or None.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = sorted(
        [c for c in contours if cv2.contourArea(c) > min_frac * img_area],
        key=cv2.contourArea, reverse=True,
    )

    for contour in candidates[:5]:
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)

        # Gradually loosen epsilon until we collapse to a quadrilateral
        for eps in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                corners = order_points(approx.reshape(4, 2)).tolist()
                if _is_valid_quad(corners, img_shape):
                    return corners

        # Last resort: rotated bounding rect
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        corners = order_points(box).tolist()
        if _is_valid_quad(corners, img_shape):
            return corners

    return None


def _otsu_mask(blurred, k):
    """Dark tray on light background: invert Otsu so the tray becomes a white blob."""
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def _grabcut_mask(img):
    """
    GrabCut with explicit mask init:
    - Hard background: 5% border strip (always table/room, never tray).
    - Probable foreground: center 70% (where the tray always is).
    - Let GrabCut resolve the ambiguous middle band using color models.

    This handles cases where tray and background share a similar color (e.g. beige
    tray on white table) because GrabCut uses spatial smoothness + color mixture models,
    not just a single global threshold.
    """
    h, w = img.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)  # probable background everywhere

    # Centre rectangle = probable foreground
    cx, cy = int(0.15 * w), int(0.15 * h)
    mask[cy: h - cy, cx: w - cx] = cv2.GC_PR_FGD

    # Thin border = certain background (gives GrabCut a clean anchor)
    bx, by = int(0.04 * w), int(0.04 * h)
    mask[:by, :] = cv2.GC_BGD
    mask[h - by:, :] = cv2.GC_BGD
    mask[:, :bx] = cv2.GC_BGD
    mask[:, w - bx:] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)


def _canny_mask(blurred, k):
    edges = cv2.Canny(blurred, 30, 100)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)


def detect_tray_corners(img, debug_path=None):
    """
    Detect the 4 corners of the thali tray.

    Tries three strategies in order, stopping at the first valid result:
      1. Otsu threshold (fast; works when tray is darker than background).
      2. GrabCut with mask init (slower; works when tray and background share
         similar color, e.g. beige tray on white table).
      3. Canny edges (fallback for high-contrast rims).

    Each strategy's result is validated: detections that span the whole image
    (corners at the image boundary) are rejected and the next strategy is tried.

    Returns [TL, TR, BR, BL] in original image coordinates, or None.
    """
    orig_h, orig_w = img.shape[:2]
    scale = DETECT_W / orig_w
    small = cv2.resize(img, (DETECT_W, int(orig_h * scale)))
    sh, sw = small.shape[:2]
    small_area = sh * sw

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    corners = None

    # Strategy 1: Otsu
    corners = _quad_from_mask(_otsu_mask(blurred, k), small_area, small.shape)

    # Strategy 2: GrabCut
    if corners is None:
        gc_mask = _grabcut_mask(small)
        if gc_mask is not None:
            corners = _quad_from_mask(gc_mask, small_area, small.shape)

    # Strategy 3: Canny
    if corners is None:
        corners = _quad_from_mask(_canny_mask(blurred, k), small_area, small.shape)

    if corners is None:
        return None

    # Scale back to original image coordinates
    corners = [[x / scale, y / scale] for x, y in corners]

    if debug_path:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        dbg = img.copy()
        pts = np.int32(corners)
        cv2.polylines(dbg, [pts], isClosed=True, color=(0, 255, 0),
                      thickness=max(3, orig_w // 300))
        font_scale = orig_w / 2000
        for label, pt in zip(["TL", "TR", "BR", "BL"], corners):
            p = (int(pt[0]), int(pt[1]))
            r = max(10, orig_w // 200)
            cv2.circle(dbg, p, r, (0, 0, 255), -1)
            cv2.putText(dbg, label, (p[0] + r + 5, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(debug_path, dbg)

    return corners


def foreshortening_ratio(corners):
    """top_edge / bottom_edge. Close to 1.0 = near top-down. Below 0.4 = very shallow."""
    tl, tr, br, bl = [np.array(p) for p in corners]
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    return top / bottom if bottom > 0 else 0.0


def estimate_output_dims(corners, base_size=700):
    """
    Compute output (width, height) that matches the tray's true aspect ratio.

    Uses the geometric mean of opposite edge lengths as a perspective-aware
    proxy for the tray's real dimensions. This avoids the square-output
    distortion that appears when a 4:3 tray is forced into a 1:1 canvas.

    Note: at very shallow angles (foreshortening ratio < 0.3), the height
    estimate is unreliable because the far edge has too few pixels to measure
    accurately. The output will still be the best geometrically-honest result,
    but some stretching on the far side is unavoidable.
    """
    tl, tr, br, bl = [np.array(p, dtype=float) for p in corners]
    apparent_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    apparent_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    aspect = apparent_w / apparent_h if apparent_h > 0 else 1.0

    if aspect >= 1.0:
        return base_size, max(1, int(base_size / aspect))
    else:
        return max(1, int(base_size * aspect)), base_size


def warp_to_bev(img, corners, out_w, out_h):
    """Perspective transform: tray corners → rectified top-down view."""
    src = np.float32(corners)
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def process(image_paths, out_size, debug):
    os.makedirs(OUTPUT_BEV, exist_ok=True)

    for img_path in image_paths:
        name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[error] Cannot read {img_path}")
            continue

        stem = os.path.splitext(name)[0]
        debug_path = os.path.join(OUTPUT_DEBUG, f"{stem}_corners.jpg") if debug else None

        corners = detect_tray_corners(img, debug_path=debug_path)
        if corners is None:
            print(f"[fail]  {name} — tray not detected (run with --debug to inspect)")
            continue

        ratio = foreshortening_ratio(corners)
        out_w, out_h = estimate_output_dims(corners, out_size)

        if ratio < 0.3:
            print(f"[warn]  {name} — extreme shallow angle (ratio={ratio:.2f}). "
                  f"Far edge has too few pixels; BEV will be stretched. "
                  f"Consider re-shooting at a steeper angle.")
        elif ratio < 0.5:
            print(f"[warn]  {name} — shallow angle (ratio={ratio:.2f}), some stretching expected")

        bev = warp_to_bev(img, corners, out_w, out_h)
        bev_path = os.path.join(OUTPUT_BEV, f"{stem}_bev.jpg")
        cv2.imwrite(bev_path, bev)
        print(f"[done]  {name} → {bev_path}  "
              f"({out_w}×{out_h}, ratio={ratio:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Process a single image by filename")
    parser.add_argument("--out-size", type=int, default=700,
                        help="Output square size in pixels (default: 700)")
    parser.add_argument("--debug", action="store_true",
                        help="Save corner detection visualisation to output/debug/")
    args = parser.parse_args()

    if args.image:
        image_paths = [os.path.join(IMAGE_DIR, args.image)]
    else:
        image_paths = sorted(
            glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
            glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
            glob.glob(os.path.join(IMAGE_DIR, "*.png"))
        )

    if not image_paths:
        print(f"No images found in '{IMAGE_DIR}/'")
        return

    print(f"Processing {len(image_paths)} image(s)...")
    process(image_paths, args.out_size, args.debug)
    print("Done. Results in output/bev/")


if __name__ == "__main__":
    main()
