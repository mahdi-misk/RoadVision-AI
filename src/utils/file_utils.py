from pathlib import Path
import os

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
