import argparse
import cv2
from src.detectors.roadvision_pipeline import RoadVisionPipeline
from src.utils.fps_counter import FPSCounter
from src.utils.config_loader import load_config
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Path to exported model (.engine)")
    parser.add_argument('--camera', type=int, default=0, help="Camera index")
    args = parser.parse_args()

    config = load_config(Path("configs/jetson_config.yaml"))
    if args.model:
        config['general_model'] = args.model
        config['enable_potholes'] = False
        config['enable_stairs'] = False

    cap = cv2.VideoCapture(args.camera)
    pipeline = RoadVisionPipeline(config)
    fps_counter = FPSCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = pipeline.process_frame(frame)
        fps = fps_counter.update()
        
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Jetson Camera", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
