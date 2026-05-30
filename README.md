# RoadVision-AI

AI-powered road scene analysis system using OpenCV, YOLO, Google Colab training, and Jetson edge deployment.

## Features
- Pothole detection
- Pothole severity classification into 3 levels: `minor_pothole`, `medium_pothole`, `severe_pothole`
- Ascending and descending stairs recognition
- General object detection using pretrained YOLO models
- Image, video, and webcam inference
- Google Colab training workflows
- Jetson edge deployment with optimized TensorRT models
- Depth camera support for estimating object distance
- Safety state classification (SAFE, WARNING, DANGER)

## Project Structure
- `configs/`: Configuration files for the app, datasets, and Jetson deployment.
- `datasets/`: Storage for pothole and stairs datasets (raw and processed).
- `models/`: Storage for custom trained and exported models.
- `notebooks/`: Google Colab notebooks for model training.
- `src/`: Main source code for inference, detectors, depth processing, and safety logic.
- `training/`: Scripts for training models locally or on a powerful machine.
- `jetson/`: Scripts for deploying and optimizing models on an NVIDIA Jetson device.
- `scripts/`: Utility scripts for downloading and preparing datasets.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RoadVision-AI.git
   cd RoadVision-AI
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run basic test (General object detection):
   ```bash
   python src/main.py --mode webcam
   ```

## How to Run Locally
- Webcam detection: `python src/main.py --mode webcam`
- Image detection: `python src/main.py --mode image --source samples/images/test.jpg --output results/images/output.jpg`
- Video detection: `python src/main.py --mode video --source samples/videos/test.mp4 --output results/videos/output.mp4`

## Training on Google Colab
1. Open `notebooks/RoadVision_AI_Colab_Training.ipynb` in Google Colab.
2. Upload or download datasets (Kaggle).
3. Prepare YOLO dataset format.
4. Train the pothole and stairs models.
5. Save `best.pt` models and copy them to `models/pothole/` and `models/stairs/`.

## Datasets
The system relies on the following dataset sources (available via Kaggle):
- Kaggle pothole datasets with severity labels (e.g., `idanbaru/annotated-potholes-with-severity-levels`)
- Kaggle stairs datasets (e.g., `samuelayman/stairs`, `akshaydattatraykhare/ascending-and-descending-staircases`)

## Model Outputs
- Bounding boxes and Class labels
- Confidence scores
- Distance in meters (when depth camera is available)
- Safety state (SAFE / WARNING / DANGER)

## Jetson Deployment
- Training is done on Colab. Final inference is done on Jetson.
- Export to ONNX: `python jetson/export_to_onnx.py --model models/pothole/best.pt`
- Export to TensorRT on the Jetson device: `python jetson/export_to_tensorrt.py --model models/pothole/best.pt --fp16`
- Use FP16 when available for faster inference.
- Recommended image sizes: 320, 416, 640.
- Recommended lightweight model: YOLO nano or small version.
- Run depth camera inference: `python jetson/run_depth_camera.py --model models/exported/roadvision.engine`

## Future Improvements
- Mobile app integration
- Voice and vibration alerts
- GPS tracking
- Jetson GPIO buzzer or vibration motor support
- Real-time assistive navigation features

## Author
Mahdi Misk
