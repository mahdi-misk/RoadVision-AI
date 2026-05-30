import argparse
from ultralytics import YOLO
import time
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    sizes = [320, 416, 640]
    
    print("Loading model...")
    start_load = time.time()
    model = YOLO(args.model)
    print(f"Model loaded in {time.time() - start_load:.2f}s")
    
    print(f"{'Image Size':<15} | {'FPS':<10} | {'Avg Inference (ms)':<20}")
    print("-" * 50)
    
    for size in sizes:
        dummy_frame = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            model(dummy_frame, verbose=False)
            
        times = []
        for _ in range(50):
            start = time.time()
            model(dummy_frame, verbose=False)
            times.append((time.time() - start) * 1000)
            
        avg_time = np.mean(times)
        fps = 1000 / avg_time
        print(f"{size:<15} | {fps:<10.1f} | {avg_time:<20.2f}")

if __name__ == "__main__":
    main()
