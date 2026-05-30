import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Validate Model")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    model = YOLO(args.model)
    metrics = model.val(data=args.data)
    print(metrics)

if __name__ == "__main__":
    main()
