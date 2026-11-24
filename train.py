from pathlib import Path

from ultralytics import YOLO, settings

EPOCHS = 10
IMAGE_SIZE = 640
DATASET_CONFIG = "coco128.yaml"
BASE_MODEL = "yolov8n.pt"
RUN_NAME = "coco128-run1"

# Tell ultralytics to look for datasets in our project folder
settings.update({"datasets_dir": str(Path("datasets").resolve())})


def train() -> Path:
    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        name=RUN_NAME,
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining done. Best weights saved to: {best_weights}")
    return best_weights


if __name__ == "__main__":
    train()
