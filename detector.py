from ultralytics import YOLO
import cv2


class Detector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, image_path: str):
        results = self.model(image_path, conf=self.confidence)
        return results[0]

    def draw(self, result) -> None:
        annotated = result.plot()
        cv2.imshow("Detections", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
