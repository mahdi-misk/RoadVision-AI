#!/bin/bash
echo "Updating apt..."
sudo apt-get update

echo "Installing pip and basic dependencies..."
sudo apt-get install -y python3-pip libopencv-dev python3-opencv

echo "Installing python requirements..."
pip3 install -r ../requirements_jetson.txt

echo "Note: TensorRT is typically installed via NVIDIA JetPack."
echo "Note: For Intel RealSense, please install pyrealsense2 via the official instructions for Jetson."
