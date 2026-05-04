from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .preprocess import preprocess

_COCO_FOOD_CLASSES = {"bowl", "cup", "plate"}
_DEFAULT_WEIGHTS = "models/thali_detector.pt"


class ThaliDetector:
    """
    Three-stage detector for thali compartments:

    1. YOLO  — fast, accurate when fine-tuned
    2. Contour fallback  — finds rectangular regions using edge detection
    3. Colour fallback   — k-means segments food by colour

    Each stage only runs if the previous one returned nothing.
    Crops for classification are always taken from the ORIGINAL image.
    """

    def __init__(
        self,
        weights: str = _DEFAULT_WEIGHTS,
        target_classes: set[str] | None = None,
        conf: float = 0.05,
        nms_iou: float = 0.25,
        max_box_area: float = 0.25,
        min_box_area: float = 0.01,
        min_aspect: float = 0.25,
        max_aspect: float = 4.0,
        mean_shift_sp: int = 20,
        mean_shift_sr: int = 40,
        use_clahe: bool = False,
        use_unsharp: bool = False,
    ):
        self.model = YOLO(weights)
        # Auto-select class filter: fine-tuned models (contain "compartment") accept
        # all their own classes; pretrained COCO models filter to food-related classes only.
        if target_classes is not None:
            self.target_classes = target_classes
        elif "compartment" in self.model.names.values():
            self.target_classes = None          # accept all — model only knows compartments
        else:
            self.target_classes = _COCO_FOOD_CLASSES
        self.conf = conf
        self.nms_iou = nms_iou
        self.max_box_area = max_box_area
        self.min_box_area = min_box_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.ms_sp = mean_shift_sp
        self.ms_sr = mean_shift_sr
        self.use_clahe = use_clahe
        self.use_unsharp = use_unsharp

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> list[dict]:
        original = cv2.imread(str(image_path))
        if original is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        smoothed = self._preprocess(original)
        image_area = original.shape[0] * original.shape[1]

        # Stage 1 — YOLO
        boxes = self._nms(self._filter(self._run_yolo(smoothed), image_area))
        if boxes:
            return boxes

        # Stage 2 — contour detection
        print("  [fallback-1] trying contour detection")
        boxes = self._contour_detect(smoothed, image_area)
        if boxes:
            return boxes

        # Stage 3 — colour segmentation
        print("  [fallback-2] trying colour segmentation")
        return self._colour_detect(smoothed, image_area)

    # ------------------------------------------------------------------
    # Stage 1 — YOLO
    # ------------------------------------------------------------------

    def _run_yolo(self, bgr: np.ndarray) -> list[dict]:
        results = self.model(bgr, conf=self.conf, iou=1.0, verbose=False)[0]
        # None = accept every class the model knows (fine-tuned model)
        # explicit set = filter to those classes only (pretrained COCO model)
        allowed = self.target_classes
        boxes = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if allowed is not None and cls_name not in allowed:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "coco_label": cls_name,
                "yolo_conf": float(box.conf),
            })
        return boxes

    def _filter(self, boxes: list[dict], image_area: int) -> list[dict]:
        kept = []
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            bw, bh = x2 - x1, y2 - y1
            if bh == 0:
                continue
            if not (self.min_box_area <= (bw * bh) / image_area <= self.max_box_area):
                continue
            if not (self.min_aspect <= bw / bh <= self.max_aspect):
                continue
            kept.append(b)
        return kept

    # ------------------------------------------------------------------
    # Stage 2 — contour detection
    # ------------------------------------------------------------------

    def _contour_detect(self, smoothed: np.ndarray, image_area: int) -> list[dict]:
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

        # Auto Canny thresholds based on image median — robust to exposure variation
        v = np.median(gray)
        lo = int(max(0,   0.67 * v))
        hi = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, lo, hi)

        # Close small gaps in compartment walls
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # RETR_LIST gets all contours, not just the outer boundary
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_box_area <= area / image_area <= self.max_box_area):
                continue

            # Approximate to polygon — must be roughly rectangular (4-8 sides)
            eps = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if not (4 <= len(approx) <= 8):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue

            # Fill ratio: contour area vs bounding rect area — rejects blobs and L-shapes
            if area / (bw * bh) < 0.45:
                continue

            if not (self.min_aspect <= bw / bh <= self.max_aspect):
                continue

            boxes.append({
                "bbox": [x, y, x + bw, y + bh],
                "coco_label": "region",
                "yolo_conf": float(area / (bw * bh)),  # fill ratio as pseudo-score
            })

        return self._nms(boxes)

    # ------------------------------------------------------------------
    # Stage 3 — colour segmentation (k-means)
    # ------------------------------------------------------------------

    def _colour_detect(self, smoothed: np.ndarray, image_area: int) -> list[dict]:
        h, w = smoothed.shape[:2]

        # Cluster in LAB space — perceptually uniform, better colour separation
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab.reshape(-1, 3)

        k = 10  # more clusters than compartments so background gets its own cluster
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(h, w).astype(np.int32)

        boxes = []
        for label_id in range(k):
            mask = np.where(labels == label_id, 255, 0).astype(np.uint8)

            # Morphological cleanup to remove noise within each colour cluster
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if not (self.min_box_area <= area / image_area <= self.max_box_area):
                    continue
                x1 = stats[i, cv2.CC_STAT_LEFT]
                y1 = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                if bh == 0:
                    continue
                if not (self.min_aspect <= bw / bh <= self.max_aspect):
                    continue
                boxes.append({
                    "bbox": [x1, y1, x1 + bw, y1 + bh],
                    "coco_label": "region",
                    "yolo_conf": 0.4,
                })

        return self._nms(boxes)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        img = preprocess(bgr, sp=self.ms_sp, sr=self.ms_sr, use_clahe=self.use_clahe)
        if self.use_unsharp:
            blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
            img = cv2.addWeighted(img, 2.5, blurred, -1.5, 0)
        return img

    def _nms(self, boxes: list[dict]) -> list[dict]:
        if not boxes:
            return []
        bboxes = torch.tensor([b["bbox"] for b in boxes], dtype=torch.float32)
        scores = torch.tensor([b["yolo_conf"] for b in boxes])
        order = scores.argsort(descending=True).tolist()
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            order = [j for j in order if _iou(bboxes[i], bboxes[j]) < self.nms_iou]
        return [boxes[i] for i in keep]


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = a.tolist()
    bx1, by1, bx2, by2 = b.tolist()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
