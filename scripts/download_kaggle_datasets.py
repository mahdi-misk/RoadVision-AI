import os

POTHOLE_DATASET = "idanbaru/annotated-potholes-with-severity-levels"
STAIRS_DATASET_1 = "samuelayman/stairs"
STAIRS_DATASET_2 = "akshaydattatraykhare/ascending-and-descending-staircases"

def main():
    print("Dataset Downloading Guide:")
    print("Method 1: Using Kaggle API (Ensure ~/.kaggle/kaggle.json is set up)")
    print(f"  kaggle datasets download -d {POTHOLE_DATASET} -p datasets/potholes/raw")
    print(f"  kaggle datasets download -d {STAIRS_DATASET_1} -p datasets/stairs/raw")
    print()
    print("Method 2: Using kagglehub in Python")
    print("  import kagglehub")
    print(f"  path = kagglehub.dataset_download('{POTHOLE_DATASET}')")
    print("  print('Path to dataset files:', path)")
    
if __name__ == "__main__":
    main()
