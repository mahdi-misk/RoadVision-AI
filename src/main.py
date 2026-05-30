import argparse
from pathlib import Path
import os

from src.utils.config_loader import load_config
from src import camera_demo
from src import image_demo
from src import video_demo

def parse_args():
    parser = argparse.ArgumentParser(description="RoadVision-AI Pipeline")
    parser.add_argument('--mode', type=str, choices=['webcam', 'image', 'video'], required=True, help="Mode to run")
    parser.add_argument('--source', type=str, help="Path to input image or video")
    parser.add_argument('--output', type=str, help="Path to save output")
    
    parser.add_argument('--objects', action='store_true', help="Enable general objects")
    parser.add_argument('--potholes', action='store_true', help="Enable potholes")
    parser.add_argument('--stairs', action='store_true', help="Enable stairs")
    parser.add_argument('--depth', action='store_true', help="Enable depth")
    
    parser.add_argument('--conf', type=float, help="Confidence threshold")
    parser.add_argument('--model', type=str, help="General model path")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = Path("configs/app_config.yaml")
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {}
        
    if args.objects: config['enable_general_objects'] = True
    if args.potholes: config['enable_potholes'] = True
    if args.stairs: config['enable_stairs'] = True
    if args.depth: config['enable_depth'] = True
    if args.conf is not None: config['confidence_threshold'] = args.conf
    if args.model is not None: config['general_model'] = args.model

    if args.mode == 'webcam':
        camera_demo.main(config)
    elif args.mode == 'image':
        if not args.source:
            print("Error: --source required for image mode")
            return
        image_demo.main(args.source, args.output, config)
    elif args.mode == 'video':
        if not args.source:
            print("Error: --source required for video mode")
            return
        video_demo.main(args.source, args.output, config)

if __name__ == "__main__":
    main()
