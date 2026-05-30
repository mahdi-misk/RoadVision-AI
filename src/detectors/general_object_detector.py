from ultralytics import YOLO
from pathlib import Path

class GeneralObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.35):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = int(box.cls)
            conf = float(box.conf)
            class_name = self.model.names[c]
            detections.append({
                'box': b,
                'class_name': class_name,
                'confidence': conf
            })
        return detections
