import argparse

from dataset import ensure_dataset, explore
from detector import Detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="sample.jpg")
    parser.add_argument("--explore", action="store_true", help="Browse the dataset")
    args = parser.parse_args()

    if args.explore:
        dataset_dir = ensure_dataset()
        explore(dataset_dir)
    else:
        detector = Detector()
        result = detector.detect(args.image)

        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)
            coords = box.xyxy[0].tolist()
            print(f"{class_name}: {confidence:.2f} @ {[round(c) for c in coords]}")

        detector.draw(result)


if __name__ == "__main__":
    main()
