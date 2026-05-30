from pathlib import Path
import argparse
import random
import shutil

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

CLASS_NAMES = {
    0: "ascending_stairs",
    1: "descending_stairs",
}


def find_direction_folders(raw_dir: Path):
    ascending_dirs = []
    descending_dirs = []

    for folder in raw_dir.rglob("*"):
        if not folder.is_dir():
            continue

        name = folder.name.lower()

        if "ascending" in name or "ascend" in name:
            ascending_dirs.append(folder)

        if "descending" in name or "descend" in name:
            descending_dirs.append(folder)

    return ascending_dirs, descending_dirs


def collect_images(folders, class_id):
    samples = []

    for folder in folders:
        for image_path in folder.rglob("*"):
            if image_path.is_file() and image_path.suffix in IMAGE_EXTENSIONS:
                samples.append((image_path, class_id))

    return samples


def make_dirs(out_dir: Path):
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_dir: Path):
    yaml_text = f"""path: {out_dir.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: ascending_stairs
  1: descending_stairs
"""
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def copy_samples(samples, out_dir: Path, split_name: str):
    for image_path, class_id in samples:
        safe_name = image_path.name.replace(" ", "_").replace("(", "").replace(")", "")
        target_image = out_dir / "images" / split_name / safe_name
        target_label = out_dir / "labels" / split_name / f"{Path(safe_name).stem}.txt"

        shutil.copy2(image_path, target_image)

        # Full-image bounding box.
        # This is a practical starting point because the ascending/descending dataset is image-level.
        # Later, for better accuracy, we can manually annotate exact stair bounding boxes.
        label_line = f"{class_id} 0.500000 0.500000 1.000000 1.000000"
        target_label.write_text(label_line + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare ascending/descending stairs dataset in YOLO format.")
    parser.add_argument("--raw", default="datasets/stairs/raw/ascending_descending", help="Raw ascending/descending stairs folder.")
    parser.add_argument("--out", default="datasets/stairs/processed", help="Output YOLO dataset folder.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.10, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clean", action="store_true", help="Clean output folder before processing.")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw stairs folder not found: {raw_dir}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    make_dirs(out_dir)

    ascending_dirs, descending_dirs = find_direction_folders(raw_dir)

    print("Found ascending folders:")
    for folder in ascending_dirs:
        print(f"  {folder}")

    print("Found descending folders:")
    for folder in descending_dirs:
        print(f"  {folder}")

    samples = []
    samples += collect_images(ascending_dirs, 0)
    samples += collect_images(descending_dirs, 1)

    if not samples:
        raise RuntimeError("No ascending/descending stair images found.")

    random.seed(args.seed)
    random.shuffle(samples)

    total = len(samples)
    test_count = int(total * args.test)
    val_count = int(total * args.val)
    train_count = total - val_count - test_count

    train_samples = samples[:train_count]
    val_samples = samples[train_count:train_count + val_count]
    test_samples = samples[train_count + val_count:]

    copy_samples(train_samples, out_dir, "train")
    copy_samples(val_samples, out_dir, "val")
    copy_samples(test_samples, out_dir, "test")

    write_data_yaml(out_dir)

    class_counts = {0: 0, 1: 0}
    for _, class_id in samples:
        class_counts[class_id] += 1

    print("Done preparing stairs dataset.")
    print(f"Total samples: {total}")
    print(f"Train: {len(train_samples)}")
    print(f"Val: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")
    print("Class counts:")
    print(f"  0 - ascending_stairs: {class_counts[0]}")
    print(f"  1 - descending_stairs: {class_counts[1]}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
