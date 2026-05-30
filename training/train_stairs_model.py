import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train Stairs Model")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data='configs/stairs_data.yaml', 
                epochs=args.epochs, 
                imgsz=args.imgsz, 
                batch=args.batch,
                project='runs/stairs_training',
                name='stairs_model')

if __name__ == "__main__":
    main()
