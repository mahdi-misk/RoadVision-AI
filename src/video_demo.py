import cv2
from src.detectors.roadvision_pipeline import RoadVisionPipeline
from src.utils.video_utils import get_video_writer
from src.utils.fps_counter import FPSCounter
import os

def main(video_path, output_path, config):
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    pipeline = RoadVisionPipeline(config)
    fps_counter = FPSCounter()
    
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = get_video_writer(output_path, fps, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, _ = pipeline.process_frame(frame)
        
        fps = fps_counter.update()
        if config.get('show_fps', True):
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if writer:
            writer.write(annotated_frame)
            
        cv2.imshow("RoadVision-AI Video Demo", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved result to {output_path}")
    cv2.destroyAllWindows()
