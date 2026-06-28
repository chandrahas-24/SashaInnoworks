import shutil
import random
from pathlib import Path

DINO_DIR  = Path("/Users/chandrahass/PycharmProjects/SashaInnoworks/yolo_dataset_dino")
PIEDS_DIR = Path("/Users/chandrahass/PycharmProjects/SashaInnoworks/pieds_15_class")
OUT_DIR   = Path("/Users/chandrahass/PycharmProjects/SashaInnoworks/yolo_final")

# Final combined class list — ORDER MATTERS (index = class id in label files)
DINO_CLASSES = [
    "Carrot", "Toaster", "Book_pack", "Ceramic_mug", "Cups", "Cutlery",
    "Glass_bottle", "Water_bottle", "Almond_pouch", "Cloth_roll", "Detergent_pack",
    "Flour_sack", "Medicine_box", "Oil_pouch", "Plastic_container", "Rice_bag",
    "Snack_packet", "Soap_bar_pack", "Spice_pouch", "Spices", "Sugar_bag",
    "Toy_box", "Yogurt", "Blueberry", "Cherry", "Custard_apple", "Dragon_fruit",
    "Garlic", "Guava", "Kiwi", "Lychee", "Mango", "Papaya", "Plum",
    "Cereal", "Electronics_item", "Grater", "Kettle"
]

PIEDS_CLASSES = [
    "Grape", "Pomegranate", "Strawberry", "Watermelon", "Pear", "Pineapple",
    "Apple", "Banana", "Orange", "Coconut", "Blender", "Bread", "Cheese",
    "Box", "Tin_can"
]

ALL_CLASSES = DINO_CLASSES + PIEDS_CLASSES  # indices 0-37 = dino, 38-52 = pieds

for split in ["train", "val"]:
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_pair(img_path, label_path, split, prefix):
    shutil.copy(img_path, OUT_DIR / "images" / split / f"{prefix}_{img_path.name}")
    if label_path.exists():
        shutil.copy(label_path, OUT_DIR / "labels" / split / f"{prefix}_{img_path.stem}.txt")

# --- yolo_dataset_dino: class ids 0-37, already correct ---
for cls in DINO_CLASSES:
    for dino_split, out_split in [("train", "train"), ("valid", "val")]:
        img_dir   = DINO_DIR / cls / dino_split / "images"
        label_dir = DINO_DIR / cls / dino_split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.glob("*.jpg"):
            copy_pair(img, label_dir / (img.stem + ".txt"), out_split, cls)

# --- pieds_15_class: class ids need remapping from 0-14 → 38-52 ---
PIEDS_OFFSET = len(DINO_CLASSES)  # 38

img_dir   = PIEDS_DIR / "images" / "val"
label_dir = PIEDS_DIR / "labels" / "val"

images = list(img_dir.glob("*.jpg"))
random.shuffle(images)
split_idx = int(len(images) * 0.8)
splits = {"train": images[:split_idx], "val": images[split_idx:]}

for out_split, imgs in splits.items():
    for img in imgs:
        # Remap label class ids
        src_label = label_dir / (img.stem + ".txt")
        dst_label = OUT_DIR / "labels" / out_split / f"pieds_{img.stem}.txt"

        if src_label.exists():
            with open(src_label) as f:
                lines = f.readlines()
            with open(dst_label, "w") as f:
                for line in lines:
                    parts = line.strip().split()
                    parts[0] = str(int(parts[0]) + PIEDS_OFFSET)  # remap id
                    f.write(" ".join(parts) + "\n")

        shutil.copy(img, OUT_DIR / "images" / out_split / f"pieds_{img.name}")

print("✅ Merge done")
for split in ["train", "val"]:
    n = len(list((OUT_DIR / "images" / split).glob("*.jpg")))
    print(f"  {split}: {n} images")

# Write yaml
with open(OUT_DIR / "dataset.yaml", "w") as f:
    f.write("path: /content/yolo_final\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write(f"nc: {len(ALL_CLASSES)}\n")
    f.write(f"names: {ALL_CLASSES}\n")

print(f"✅ dataset.yaml written — {len(ALL_CLASSES)} classes total")