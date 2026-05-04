from pathlib import Path
from PIL import Image

from .classifier import KhanaClassifier
from .detector import ThaliDetector


class ThaliPipeline:
    """Chains detection → crop → Khana classification for a single image."""

    def __init__(
        self,
        classifier: KhanaClassifier,
        detector: ThaliDetector,
        pad: int = 10,
    ):
        self.classifier = classifier
        self.detector = detector
        self.pad = pad

    def run(self, image_path: str | Path) -> list[dict]:
        """
        Returns list of dicts per detected food item:
            bbox        : [x1, y1, x2, y2]
            label       : Khana class name
            conf        : classifier confidence
            coco_label  : original YOLO label (for debugging)
        """
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        detections = self.detector.detect(str(image_path))

        if not detections:
            print(f"[warn] no regions passed filters in {Path(image_path).name}")
            return []

        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Pad and clamp to image bounds
            x1, y1 = max(0, x1 - self.pad), max(0, y1 - self.pad)
            x2, y2 = min(w, x2 + self.pad), min(h, y2 + self.pad)

            crop = image.crop((x1, y1, x2, y2))
            label, conf = self.classifier.predict(crop)

            results.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": conf,
                "coco_label": det["coco_label"],
            })

        return results
