from ultralytics import YOLO
from pathlib import Path
import os

class StairsDetector:
    def __init__(self, model_path="models/stairs/best.pt", conf_threshold=0.35):
        self.conf_threshold = conf_threshold
        if not os.path.exists(model_path):
            print("Custom stairs model not found. Train the model first using the Colab notebook, then place best.pt inside models/stairs/")
            self.model = None
        else:
            self.model = YOLO(model_path)

    def detect(self, frame):
        if self.model is None:
            return []
            
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
