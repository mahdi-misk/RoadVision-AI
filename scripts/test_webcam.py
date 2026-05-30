import cv2
from ultralytics import YOLO

def main():
    print("Loading pothole model... Please wait.")
    
    # Load trained pothole model
    pothole_model = YOLO("models/pothole/pothole_yolov8_final.pt")
    
    print("Model loaded successfully! Starting webcam...")

    # Start webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam is active. Press 'q' on your keyboard to close the window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Predict potholes (with at least 50% confidence)
        pothole_results = pothole_model.predict(source=frame, conf=0.5, verbose=False)
        
        # Draw bounding boxes for potholes
        annotated_frame = pothole_results[0].plot()
        
        # Show final result
        cv2.imshow("RoadVision-AI Live Inference", annotated_frame)

        # Close window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
