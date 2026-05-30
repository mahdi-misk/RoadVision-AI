import os

folders = [
    'configs',
    'datasets/potholes/raw',
    'datasets/potholes/processed',
    'datasets/stairs/raw',
    'datasets/stairs/processed',
    'models/pothole',
    'models/stairs',
    'models/general',
    'models/exported',
    'notebooks',
    'src/detectors',
    'src/depth',
    'src/safety',
    'src/utils',
    'src/assets',
    'training',
    'jetson',
    'scripts',
    'samples/images',
    'samples/videos',
    'results/images',
    'results/videos',
    'results/logs',
]

def main():
    for f in folders:
        os.makedirs(f, exist_ok=True)
    print("Folders created successfully.")

if __name__ == "__main__":
    main()
