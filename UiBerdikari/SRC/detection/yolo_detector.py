from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, task="detect", confidence=0.65):
        self.model = YOLO(model_path, task = task)
        self.confidence = confidence

    def predict(self, frame):
        return self.model(frame, verbose=False, conf=self.confidence)
