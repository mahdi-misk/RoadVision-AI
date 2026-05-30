import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='datasets/stairs/raw')
    parser.add_argument('--output', default='datasets/stairs/processed')
    args = parser.parse_args()

    # Create train/val/test folders
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'labels', split), exist_ok=True)
        
    print(f"Created YOLO dataset structure at {args.output}")
    print("Note: Actual conversion logic depends on the specific annotation format of the downloaded dataset.")
    print("Classes: ascending_stairs, descending_stairs")
    print("We do not create fake labels.")

if __name__ == "__main__":
    main()
