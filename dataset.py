from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.downloads import download

DATASET_URL = "https://ultralytics.com/assets/coco128.zip"
DATASET_DIR = Path("datasets/coco128")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def ensure_dataset() -> Path:
    """Download coco128 if not already present."""
    if not DATASET_DIR.exists():
        print("Downloading coco128 (~7MB)...")
        download(DATASET_URL, dir=Path("datasets"))
    return DATASET_DIR


def parse_label(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """
    Parse a YOLO-format .txt annotation file.

    Each line contains 5 numbers:
        class_id  x_center  y_center  width  height

    All coordinates are normalized to [0, 1] relative to image size.
    We convert them back to pixel coordinates for drawing.
    """
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        class_id = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:])

        # Convert: center + size (normalized) → top-left + bottom-right (pixels)
        x1 = int((x_c - w / 2) * img_w)
        y1 = int((y_c - h / 2) * img_h)
        x2 = int((x_c + w / 2) * img_w)
        y2 = int((y_c + h / 2) * img_h)

        boxes.append({"class_id": class_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return boxes


def draw_boxes(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Draw ground-truth boxes (green) on a copy of the image."""
    result = image.copy()
    for box in boxes:
        name = COCO_CLASSES[box["class_id"]]
        cv2.rectangle(result, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2)
        cv2.putText(result, name, (box["x1"], box["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result


def explore(dataset_dir: Path) -> None:
    """Browse dataset images one by one. Any key → next image. Q → quit."""
    images_dir = dataset_dir / "images" / "train2017"
    labels_dir = dataset_dir / "labels" / "train2017"

    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"{len(image_paths)} images. Press any key to advance, Q to quit.")

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        label_path = labels_dir / img_path.with_suffix(".txt").name
        boxes = parse_label(label_path, w, h)

        annotated = draw_boxes(image, boxes)
        cv2.imshow(f"{img_path.name}  |  {len(boxes)} objects", annotated)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
