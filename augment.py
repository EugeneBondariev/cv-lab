from pathlib import Path

import albumentations as A
import cv2

from dataset import COCO_CLASSES, DATASET_DIR, draw_boxes, parse_label

OUTPUT_DIR = Path("augment_output")
NUM_SAMPLES = 5
AUGMENTATIONS_PER_IMAGE = 4

# Each transform has a probability p of being applied.
# bbox_params tells albumentations to transform bounding boxes along with the image.
pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Rotate(limit=10, p=0.4),
        A.RandomCrop(height=400, width=400, p=0.5),
    ],
    bbox_params=A.BboxParams(
        format="yolo",          # normalized x_center, y_center, width, height
        label_fields=["class_ids"],
        min_visibility=0.3,     # drop a box if augmentation hides >70% of it
    ),
)


def load_yolo_bboxes(label_path: Path) -> tuple[list, list]:
    """Return (bboxes, class_ids) in YOLO normalized format."""
    bboxes, class_ids = [], []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            class_ids.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:]])
    return bboxes, class_ids


def yolo_to_pixel_boxes(bboxes: list, class_ids: list, w: int, h: int) -> list[dict]:
    """Convert normalized YOLO boxes back to pixel dicts for draw_boxes."""
    boxes = []
    for (x_c, y_c, bw, bh), cid in zip(bboxes, class_ids):
        boxes.append({
            "class_id": int(cid),
            "x1": int((x_c - bw / 2) * w),
            "y1": int((y_c - bh / 2) * h),
            "x2": int((x_c + bw / 2) * w),
            "y2": int((y_c + bh / 2) * h),
        })
    return boxes


def augment_and_save(img_path: Path, label_path: Path) -> None:
    image_bgr = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # albumentations expects RGB

    bboxes, class_ids = load_yolo_bboxes(label_path)

    for i in range(AUGMENTATIONS_PER_IMAGE):
        result = pipeline(image=image_rgb, bboxes=bboxes, class_ids=class_ids)

        aug_image = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        h, w = aug_image.shape[:2]
        boxes = yolo_to_pixel_boxes(result["bboxes"], result["class_ids"], w, h)

        annotated = draw_boxes(aug_image, boxes)
        out_path = OUTPUT_DIR / f"{img_path.stem}_aug{i + 1}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    images_dir = DATASET_DIR / "images" / "train2017"
    labels_dir = DATASET_DIR / "labels" / "train2017"
    image_paths = sorted(images_dir.glob("*.jpg"))[:NUM_SAMPLES]

    for img_path in image_paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        augment_and_save(img_path, label_path)

    print(f"\nDone. {NUM_SAMPLES * AUGMENTATIONS_PER_IMAGE} images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
