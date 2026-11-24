import sys
from detector import Detector


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"

    detector = Detector()
    result = detector.detect(image_path)

    for box in result.boxes:
        class_id = int(box.cls)
        class_name = result.names[class_id]
        confidence = float(box.conf)
        coords = box.xyxy[0].tolist()
        print(f"{class_name}: {confidence:.2f} @ {[round(c) for c in coords]}")

    detector.draw(result)


if __name__ == "__main__":
    main()
