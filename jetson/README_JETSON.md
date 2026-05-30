# Jetson Deployment Guide

## Hardware target
- NVIDIA Jetson Nano
- NVIDIA Jetson Orin Nano Super

## Software requirements
- JetPack (which includes TensorRT)
- Python 3
- OpenCV
- Ultralytics YOLO

## Setup steps
1. Clone the repo on Jetson.
2. Run `./install_jetson_dependencies.sh` to install basics.

## Copy model to Jetson
Train on Colab -> Download `best.pt` -> Copy to `RoadVision-AI/models/pothole/best.pt`.

## Export to ONNX
```bash
python jetson/export_to_onnx.py --model models/pothole/best.pt
```

## Export to TensorRT
Execute this directly on the Jetson!
```bash
python jetson/export_to_tensorrt.py --model models/pothole/best.pt --fp16
```

## Run USB camera
```bash
python jetson/run_jetson_camera.py --model models/exported/best.engine --camera 0
```

## Run depth camera
```bash
python jetson/run_depth_camera.py --model models/exported/best.engine
```

## Benchmark FPS
```bash
python jetson/benchmark_jetson.py --model models/exported/best.engine
```

## Performance tips
- Use YOLO nano or small.
- Use image size 416 first.
- Use TensorRT FP16.
- Disable unused detectors if FPS is low.
- Export TensorRT engine directly on Jetson.
