import cv2
import torch
import numpy as np
from ultralytics import YOLO

def main():
    print("Loading AI Models... This might take a minute.")

    # 1. Load YOLO Models
    print("- Loading YOLOv8 Pothole Model...")
    pothole_model = YOLO("models/pothole/pothole_yolov8_final.pt")
    
    print("- Loading YOLOv8 Stairs Model...")
    stairs_model = YOLO("models/stairs/stairs_yolov8_final.pt")

    print("- Loading YOLOv8 General Obstacles Model (Cars, People, etc.)...")
    obstacle_model = YOLO("yolov8n.pt")

    # 2. Load MiDaS Depth Estimation Model
    print("- Loading MiDaS Depth Estimation Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()

    # Load MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print("✅ All models loaded successfully!")

    # 3. Start Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 4. Generate Depth Map using MiDaS
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            # Resize depth map to original frame size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)

        # 5. Run YOLO Predictions
        pothole_results = pothole_model.predict(source=frame, conf=0.5, verbose=False)[0]
        stairs_results = stairs_model.predict(source=frame, conf=0.5, verbose=False)[0]
        # We use a slightly higher confidence for general obstacles to reduce noise
        obstacle_results = obstacle_model.predict(source=frame, conf=0.6, verbose=False)[0]

        annotated_frame = frame.copy()

        # Helper function to process detections
        def process_detections(results, class_names):
            for box in results.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]

                # Extract depth for this bounding box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
                
                box_depth_values = depth_map[y1:y2, x1:x2]
                
                if box_depth_values.size == 0:
                    continue

                median_inverse_depth = np.median(box_depth_values)
                
                # Simulate Metric Distance (Meters)
                if median_inverse_depth > 0:
                    simulated_distance = 5000.0 / median_inverse_depth
                else:
                    simulated_distance = 99.9

                # Determine Safety State
                if simulated_distance > 5.0:
                    state = "SAFE"
                    color = (0, 255, 0) # Green
                elif 2.0 <= simulated_distance <= 5.0:
                    state = "WARNING"
                    color = (0, 255, 255) # Yellow
                else:
                    state = "DANGER"
                    color = (0, 0, 255) # Red

                # Draw Bounding Box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw Label
                label = f"{class_name} | {state} | {simulated_distance:.1f}m"
                
                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                
                # Label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Process Potholes
        pothole_class_names = pothole_results.names
        process_detections(pothole_results, pothole_class_names)

        # Process Stairs
        stairs_class_names = stairs_results.names
        process_detections(stairs_results, stairs_class_names)

        # Process General Obstacles (Cars, People, etc.)
        obstacle_class_names = obstacle_results.names
        process_detections(obstacle_results, obstacle_class_names)

        # Show Output side-by-side
        combined_view = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow("RoadVision-AI: Live Inference & Depth", combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
