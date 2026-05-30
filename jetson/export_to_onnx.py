import argparse
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to .pt model")
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.model)
    output = model.export(format='onnx', imgsz=args.imgsz, simplify=True)
    
    os.makedirs('models/exported', exist_ok=True)
    print(f"Exported to {output}. Move it to models/exported/ to keep repo organized.")

if __name__ == "__main__":
    main()
