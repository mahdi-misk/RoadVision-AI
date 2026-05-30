import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'engine'])
    args = parser.parse_args()

    model = YOLO(args.model)
    print(f"Exporting model to {args.format}...")
    if args.format == 'engine':
        print("Note: TensorRT engine files should be exported on the same Jetson device where they will run because engine files are hardware and environment dependent.")
    
    model.export(format=args.format)
    print("Export complete. Please move the exported model to models/exported/")

if __name__ == "__main__":
    main()
