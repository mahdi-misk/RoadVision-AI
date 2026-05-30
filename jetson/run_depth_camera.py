import argparse
import cv2
from src.detectors.roadvision_pipeline import RoadVisionPipeline
from src.depth.realsense_depth import RealSenseCamera
from src.utils.config_loader import load_config
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    config = load_config(Path("configs/jetson_config.yaml"))
    config['enable_depth'] = True
    if args.model:
        config['general_model'] = args.model

    dc = RealSenseCamera()
    if not dc.start():
        print("Failed to start depth camera.")
        return

    pipeline = RoadVisionPipeline(config)

    while True:
        ret, color_frame, depth_frame = dc.read()
        if not ret:
            continue

        annotated, _ = pipeline.process_frame(color_frame, depth_frame)
        
        cv2.imshow("Jetson Depth Camera", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dc.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
