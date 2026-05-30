import cv2
from src.detectors.roadvision_pipeline import RoadVisionPipeline
import os

def main(image_path, output_path, config):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    frame = cv2.imread(image_path)
    pipeline = RoadVisionPipeline(config)
    
    annotated_frame, _ = pipeline.process_frame(frame)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved result to {output_path}")
        
    cv2.imshow("RoadVision-AI Image Demo", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
