import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train Pothole Model")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data='configs/pothole_data.yaml', 
                epochs=args.epochs, 
                imgsz=args.imgsz, 
                batch=args.batch,
                project='runs/pothole_training',
                name='pothole_model')

if __name__ == "__main__":
    main()
