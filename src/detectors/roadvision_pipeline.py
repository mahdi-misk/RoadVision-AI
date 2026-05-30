import cv2
from src.detectors.general_object_detector import GeneralObjectDetector
from src.detectors.pothole_detector import PotholeDetector
from src.detectors.stairs_detector import StairsDetector
from src.depth.obstacle_distance_estimator import estimate_distances
from src.safety.safety_state import evaluate_safety
from src.utils.drawing import draw_bbox

class RoadVisionPipeline:
    def __init__(self, config):
        self.config = config
        self.detectors = []
        
        if config.get('enable_general_objects', True):
            self.detectors.append(GeneralObjectDetector(config.get('general_model', 'yolov8n.pt'), config.get('confidence_threshold', 0.35)))
        if config.get('enable_potholes', True):
            self.detectors.append(PotholeDetector(config.get('pothole_model', 'models/pothole/best.pt'), config.get('confidence_threshold', 0.35)))
        if config.get('enable_stairs', True):
            self.detectors.append(StairsDetector(config.get('stairs_model', 'models/stairs/best.pt'), config.get('confidence_threshold', 0.35)))
            
        self.enable_depth = config.get('enable_depth', False)

    def process_frame(self, frame, depth_frame=None):
        all_detections = []
        for detector in self.detectors:
            all_detections.extend(detector.detect(frame))
            
        if self.enable_depth and depth_frame is not None:
            all_detections = estimate_distances(all_detections, depth_frame)
        else:
            for det in all_detections:
                det['label'] = f"{det['class_name']} {det['confidence']:.2f}"
                det['distance_m'] = None
                
        # Draw and evaluate safety
        annotated_frame = frame.copy()
        highest_risk = "SAFE"
        
        for det in all_detections:
            safety = evaluate_safety(det['class_name'], det['confidence'], det.get('distance_m'))
            
            if safety == "DANGER":
                color = (0, 0, 255) # Red
                highest_risk = "DANGER"
            elif safety == "WARNING":
                color = (0, 255, 255) # Yellow
                if highest_risk != "DANGER":
                    highest_risk = "WARNING"
            else:
                color = (0, 255, 0) # Green
                
            draw_bbox(annotated_frame, det['box'], det['label'], color)
            
        # Draw global safety state
        cv2.putText(annotated_frame, f"STATUS: {highest_risk}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255) if highest_risk == "DANGER" else (0, 255, 255) if highest_risk == "WARNING" else (0, 255, 0), 2)
                    
        return annotated_frame, all_detections
