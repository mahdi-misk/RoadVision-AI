import cv2
import argparse
from src.detectors.roadvision_pipeline import RoadVisionPipeline
from src.utils.fps_counter import FPSCounter

def main(config):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    pipeline = RoadVisionPipeline(config)
    fps_counter = FPSCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, _ = pipeline.process_frame(frame)
        
        fps = fps_counter.update()
        if config.get('show_fps', True):
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("RoadVision-AI Webcam Demo", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
