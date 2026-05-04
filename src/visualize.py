from pathlib import Path
import cv2


def draw_detections(
    image_path: str | Path,
    detections: list[dict],
    output_path: str | Path,
) -> None:
    """Draw Khana labels + bounding boxes and save to output_path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    cv2.imwrite(str(output_path), img)
    print(f"  saved → {output_path}")
