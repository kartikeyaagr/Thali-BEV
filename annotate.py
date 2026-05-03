"""
Step 1: Annotate tray corners for each image.

Click exactly 4 corners of the thali tray (top-left, top-right,
bottom-right, bottom-left) in order. Press 'r' to reset, 'n' to skip,
's' to save and move to next image.

Saves all annotations to corners.json.
"""

import cv2
import json
import os
import glob
import numpy as np

IMAGE_DIR = "images"
OUTPUT_FILE = "corners.json"
DISPLAY_WIDTH = 900  # resize for display only, annotations are stored in original coords


def order_points(pts):
    """Order points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]    # top-left: smallest sum
    ordered[2] = pts[np.argmax(s)]    # bottom-right: largest sum
    ordered[1] = pts[np.argmin(diff)] # top-right: smallest diff
    ordered[3] = pts[np.argmax(diff)] # bottom-left: largest diff
    return ordered


def annotate_images():
    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
        glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    )

    if not image_paths:
        print(f"No images found in '{IMAGE_DIR}/'")
        return

    # Load existing annotations so we can resume
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            annotations = json.load(f)
    else:
        annotations = {}

    for img_path in image_paths:
        name = os.path.basename(img_path)
        if name in annotations:
            print(f"[skip] {name} already annotated")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[error] Could not read {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]
        scale = DISPLAY_WIDTH / orig_w
        disp = cv2.resize(img, (DISPLAY_WIDTH, int(orig_h * scale)))

        points = []  # in display coords

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(disp, (x, y), 6, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(disp, points[-2], points[-1], (0, 255, 0), 2)
                if len(points) == 4:
                    cv2.line(disp, points[-1], points[0], (0, 255, 0), 2)
                cv2.imshow("Annotate", disp)

        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", mouse_cb)

        print(f"\n[{name}] Click 4 tray corners (TL → TR → BR → BL)")
        print("  r = reset  |  n = skip  |  s = save & next")

        while True:
            cv2.imshow("Annotate", disp)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('r'):
                points.clear()
                img_fresh = cv2.imread(img_path)
                disp[:] = cv2.resize(img_fresh, (DISPLAY_WIDTH, int(orig_h * scale)))
                cv2.imshow("Annotate", disp)

            elif key == ord('n'):
                print(f"  skipped {name}")
                break

            elif key == ord('s'):
                if len(points) != 4:
                    print(f"  need exactly 4 points, have {len(points)}")
                    continue
                # Convert display coords back to original image coords
                orig_pts = [(int(x / scale), int(y / scale)) for x, y in points]
                ordered = order_points(orig_pts).tolist()
                annotations[name] = ordered
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(annotations, f, indent=2)
                print(f"  saved {name}")
                break

    cv2.destroyAllWindows()
    print(f"\nDone. Annotations saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    annotate_images()
