from pathlib import Path
import argparse
import random
import shutil
import xml.etree.ElementTree as ET


CLASS_MAP = {
    "minor_pothole": 0,
    "medium_pothole": 1,
    "major_pothole": 2,
    "severe_pothole": 2,
}

CLASS_NAMES = {
    0: "minor_pothole",
    1: "medium_pothole",
    2: "severe_pothole",
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def find_image(images_dir: Path, filename: str):
    candidate = images_dir / filename
    if candidate.exists():
        return candidate

    stem = Path(filename).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def parse_xml(xml_path: Path, images_dir: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    if filename_node is None or not filename_node.text:
        return None, []

    image_path = find_image(images_dir, filename_node.text.strip())
    if image_path is None:
        return None, []

    size = root.find("size")
    if size is None:
        return None, []

    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    labels = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        box_node = obj.find("bndbox")

        if name_node is None or box_node is None:
            continue

        class_name = name_node.text.strip()

        if class_name not in CLASS_MAP:
            print(f"[WARNING] Unknown class '{class_name}' in {xml_path.name}, skipped.")
            continue

        class_id = CLASS_MAP[class_name]

        xmin = float(box_node.find("xmin").text)
        ymin = float(box_node.find("ymin").text)
        xmax = float(box_node.find("xmax").text)
        ymax = float(box_node.find("ymax").text)

        x_center, y_center, width, height = voc_box_to_yolo(
            xmin, ymin, xmax, ymax, img_w, img_h
        )

        labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return image_path, labels


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
  0: minor_pothole
  1: medium_pothole
  2: severe_pothole
"""
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert pothole VOC XML dataset to YOLO format.")
    parser.add_argument("--raw", default="datasets/potholes/raw", help="Raw pothole dataset folder.")
    parser.add_argument("--out", default="datasets/potholes/processed", help="Output YOLO dataset folder.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.10, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clean", action="store_true", help="Clean output folder before processing.")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found: {annotations_dir}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    make_dirs(out_dir)

    xml_files = sorted(annotations_dir.glob("*.xml"))
    samples = []

    class_counts = {0: 0, 1: 0, 2: 0}
    skipped = 0

    for xml_path in xml_files:
        image_path, labels = parse_xml(xml_path, images_dir)

        if image_path is None or not labels:
            skipped += 1
            continue

        for line in labels:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

        samples.append((image_path, labels))

    random.seed(args.seed)
    random.shuffle(samples)

    total = len(samples)
    test_count = int(total * args.test)
    val_count = int(total * args.val)
    train_count = total - val_count - test_count

    splits = {
        "train": samples[:train_count],
        "val": samples[train_count:train_count + val_count],
        "test": samples[train_count + val_count:],
    }

    for split_name, split_samples in splits.items():
        for image_path, labels in split_samples:
            target_image = out_dir / "images" / split_name / image_path.name
            target_label = out_dir / "labels" / split_name / f"{image_path.stem}.txt"

            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")

    write_data_yaml(out_dir)

    print("Done converting pothole dataset to YOLO format.")
    print(f"Total XML files: {len(xml_files)}")
    print(f"Valid samples: {total}")
    print(f"Skipped samples: {skipped}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print("Class counts:")
    for class_id, count in class_counts.items():
        print(f"  {class_id} - {CLASS_NAMES[class_id]}: {count}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
