from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, settings

from dataset import COCO_CLASSES, DATASET_DIR, draw_boxes, parse_label

BEST_WEIGHTS = Path("runs/detect/coco128-run1/weights/best.pt")
OUTPUT_DIR = Path("eval_output")
NUM_SAMPLES = 8


def predictions_to_boxes(result) -> list[dict]:
    """Convert ultralytics result to the box dict format draw_boxes expects."""
    boxes = []
    for box in result.boxes:
        class_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append({
            "class_id": class_id,
            "label": f"{COCO_CLASSES[class_id]} {conf:.2f}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return boxes


def save_comparisons(model: YOLO) -> None:
    """Save side-by-side ground truth vs prediction images."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    images_dir = DATASET_DIR / "images" / "train2017"
    labels_dir = DATASET_DIR / "labels" / "train2017"
    image_paths = sorted(images_dir.glob("*.jpg"))[:NUM_SAMPLES]

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        # Left: ground truth boxes in green
        label_path = labels_dir / img_path.with_suffix(".txt").name
        gt_boxes = parse_label(label_path, w, h)
        gt_image = draw_boxes(image, gt_boxes)

        # Right: model predictions in red
        result = model(img_path, verbose=False)[0]
        pred_image = draw_boxes(image, predictions_to_boxes(result), color=(0, 0, 255))

        # Header bar with labels
        header = np.zeros((30, w * 2, 3), dtype=np.uint8)
        cv2.putText(header, "GROUND TRUTH", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(header, "PREDICTIONS", (w + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        combined = cv2.vconcat([header, cv2.hconcat([gt_image, pred_image])])
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), combined)
        print(f"Saved: {out_path}")


def print_class_metrics(val_results) -> None:
    """Print per-class mAP50-95, sorted best to worst."""
    names = val_results.names
    maps = val_results.box.maps  # per-class mAP50-95

    pairs = [(names[i], maps[i]) for i in range(len(maps))]
    pairs.sort(key=lambda x: x[1], reverse=True)

    print("\nPer-class mAP50-95:")
    print(f"{'Class':<20} {'mAP':>6}")
    print("-" * 28)
    for name, ap in pairs:
        bar = "█" * int(ap * 20)
        print(f"{name:<20} {ap:>6.3f}  {bar}")

    print(f"\nOverall mAP50:    {val_results.box.map50:.3f}")
    print(f"Overall mAP50-95: {val_results.box.map:.3f}")


def evaluate() -> None:
    settings.update({"datasets_dir": str(Path("datasets").resolve())})

    model = YOLO(BEST_WEIGHTS)

    print("Running validation...")
    val_results = model.val(data="coco128.yaml", verbose=False)
    print_class_metrics(val_results)

    print(f"\nSaving {NUM_SAMPLES} comparison images to {OUTPUT_DIR}/...")
    save_comparisons(model)
    print("Done. Open the eval_output/ folder to inspect predictions vs ground truth.")


if __name__ == "__main__":
    evaluate()
