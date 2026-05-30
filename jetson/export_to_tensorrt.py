import argparse
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to .pt model")
    parser.add_argument('--fp16', action='store_true', help="Enable FP16 precision")
    args = parser.parse_args()

    print("WARNING: TensorRT engine files should be exported on the same Jetson device where they will run because engine files are hardware and environment dependent.")

    model = YOLO(args.model)
    output = model.export(format='engine', half=args.fp16)
    
    os.makedirs('models/exported', exist_ok=True)
    print(f"Exported to {output}. Move it to models/exported/ to keep repo organized.")

if __name__ == "__main__":
    main()
