import os
import shutil
import random
import json
import yaml
import numpy as np
from PIL import Image
from pathlib import Path

# Organize raw training data for YOLO to train on.

RAW = Path("data/raw")
OUT = Path("data/yolo_dataset")
# path to folder with background images *removing hardcoded file paths soon*
BACKGROUNDS = Path(" ")

# Apply background image to objects
def composite_onto_background(render_path, output_path):
    # Paste transparent render onto a random background.
    part = Image.open(render_path).convert("RGBA")

    bg_paths = (
        list(BACKGROUNDS.glob("*.jpg")) +
        list(BACKGROUNDS.glob("*.png")) +
        list(BACKGROUNDS.glob("*.jpeg"))
    )
    bg = Image.open(random.choice(bg_paths)).convert("RGB")
    bg = bg.resize(part.size)

    bg.paste(part, (0, 0), part)
    bg.convert("RGB").save(output_path)

# Create folder structure YOLO expects
for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# One pass over every part folder in data/raw/
class_names = []

for class_id, part_folder in enumerate(sorted(RAW.iterdir())):
    if not part_folder.is_dir():
        continue

    # Read part name from metadata.json
    meta = json.load(open(part_folder / "metadata.json"))
    class_names.append(meta["name"])

    # Collect all image/label pairs
    images = sorted(part_folder.glob("*.png"))
    random.shuffle(images)
    # 80% train, 20% val
    split_idx = int(len(images) * 0.8)

    for i, img_path in enumerate(images):
        label_path = img_path.with_suffix(".txt")

        # Skip if annotation is missing
        if not label_path.exists():
            print(f"WARNING: No annotation for {img_path.name}, skipping.")
            continue

        split = "train" if i < split_idx else "val"

        output_img = OUT / "images" / split / img_path.name

        # Add random background
        composite_onto_background(img_path, output_img)

        shutil.copy(
            label_path,
            OUT / "labels" / split / label_path.name
        )

    print(f"class {class_id} — {meta['name']}: {split_idx} train, {len(images)-split_idx} val")

# Write dataset.yaml
yaml_data = {
    # absolute path to yolo_dataset/
    "path": str(OUT.resolve()),
    "train": "images/train",
    "val": "images/val",
    # number of classes
    "nc": len(class_names),
    "names": class_names,
}

with open(OUT / "dataset.yaml", "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False)

print(f"\ndataset.yaml written — {len(class_names)} classes: {class_names}")
print("Ready to train.")