import os
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter

def get_classes_from_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    return names

def main():
    # Dynamic path resolution relative to this script
    script_dir = Path(__file__).resolve().parent
    dino_dir = script_dir / "yolo_dataset_dino"
    pieds_dir = script_dir / "pieds_15_class"
    out_dir = script_dir / "yolo_final"

    print("🚀 Starting dataset merge process...")

    # Load classes
    dino_classes = get_classes_from_yaml(dino_dir / "data.yaml")
    pieds_classes = get_classes_from_yaml(pieds_dir / "dataset.yaml")
    combined_classes = dino_classes + pieds_classes
    offset = len(dino_classes)

    print(f"📁 Source Dino dataset has {len(dino_classes)} classes.")
    print(f"📁 Source Pieds dataset has {len(pieds_classes)} classes.")
    print(f"📋 Unified dataset will have {len(combined_classes)} classes.")

    # Recreate output structure cleanly
    if out_dir.exists():
        print(f"🧹 Clearing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # --- 1. Merge DINO Dataset (each class has separate subfolder) ---
    print("\n--- 1/2 Processing DINO Dataset ---")
    dino_img_copied = 0
    dino_lbl_copied = 0

    for cls in dino_classes:
        cls_idx = dino_classes.index(cls)
        # Check source class folders
        class_folder = dino_dir / cls
        if not class_folder.exists():
            print(f"⚠️  Warning: Class folder {cls} not found under {dino_dir}")
            continue

        for split in ["train", "valid"]:
            src_img_dir = class_folder / split / "images"
            src_lbl_dir = class_folder / split / "labels"
            if not src_img_dir.exists():
                continue

            out_split = "train" if split == "train" else "val"

            for img_path in src_img_dir.glob("*.jpg"):
                dst_img_name = f"{cls}_{img_path.name}"
                dst_img_path = out_dir / "images" / out_split / dst_img_name
                shutil.copy2(img_path, dst_img_path)
                dino_img_copied += 1

                # Process corresponding label
                lbl_path = src_lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    dst_lbl_name = f"{cls}_{lbl_path.name}"
                    dst_lbl_path = out_dir / "labels" / out_split / dst_lbl_name
                    
                    # Read and remap class ID (always 0) to cls_idx
                    with open(lbl_path, "r") as f:
                        lines = f.readlines()
                    
                    remapped_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            # In autodistill label output, the first value is the class index (always 0)
                            # Remap it to the actual class index
                            parts[0] = str(cls_idx)
                            remapped_lines.append(" ".join(parts))
                    
                    with open(dst_lbl_path, "w") as f:
                        f.write("\n".join(remapped_lines) + "\n")
                    dino_lbl_copied += 1

    print(f"✅ DINO dataset processed: copied {dino_img_copied} images, {dino_lbl_copied} labels.")

    # --- 2. Merge PIEDS Dataset (needs random splitting 80/20 of 'val' split) ---
    print("\n--- 2/2 Processing PIEDS Dataset ---")
    pieds_img_copied = 0
    pieds_lbl_copied = 0

    src_img_dir = pieds_dir / "images" / "val"
    src_lbl_dir = pieds_dir / "labels" / "val"

    if not src_img_dir.exists():
        raise FileNotFoundError(f"🚨 Pieds validation images directory {src_img_dir} does not exist!")

    # Sort images first to ensure deterministic shuffle with random seed
    images = sorted(list(src_img_dir.glob("*.jpg")))
    
    # 80/20 train/val split with fixed seed
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    splits = {"train": train_imgs, "val": val_imgs}
    print(f"📊 Splitting {len(images)} PIEDS images: {len(train_imgs)} for train, {len(val_imgs)} for val.")

    for out_split, imgs in splits.items():
        for img_path in imgs:
            dst_img_name = f"pieds_{img_path.name}"
            dst_img_path = out_dir / "images" / out_split / dst_img_name
            shutil.copy2(img_path, dst_img_path)
            pieds_img_copied += 1

            # Process label file and remap class ID (0-14 -> 38-52)
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                dst_lbl_name = f"pieds_{lbl_path.name}"
                dst_lbl_path = out_dir / "labels" / out_split / dst_lbl_name
                
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                
                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        orig_id = int(parts[0])
                        new_id = orig_id + offset
                        parts[0] = str(new_id)
                        remapped_lines.append(" ".join(parts))
                
                with open(dst_lbl_path, "w") as f:
                    f.write("\n".join(remapped_lines) + "\n")
                pieds_lbl_copied += 1

    print(f"✅ PIEDS dataset processed: copied {pieds_img_copied} images, {pieds_lbl_copied} labels.")

    # --- 3. Write dataset.yaml ---
    print("\n📝 Writing dataset.yaml...")
    with open(out_dir / "dataset.yaml", "w") as f:
        f.write(f"path: {out_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(combined_classes)}\n")
        f.write("names:\n")
        for idx, name in enumerate(combined_classes):
            f.write(f"  {idx}: {name}\n")
    print(f"✅ dataset.yaml written successfully with {len(combined_classes)} classes.")

    # --- 4. Validate Dataset Integrity ---
    print("\n🧐 Validating merged dataset...")
    validate_dataset(out_dir, dino_classes, combined_classes)

def validate_dataset(out_dir, dino_classes, combined_classes):
    dino_count = len(dino_classes)
    total_count = len(combined_classes)
    box_counter = Counter()

    for split in ["train", "val"]:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split

        img_files = sorted(list(img_dir.glob("*.jpg")))
        lbl_files = sorted(list(lbl_dir.glob("*.txt")))

        print(f"\n📂 Split [{split}]:")
        print(f"  Images: {len(img_files)}")
        print(f"  Labels: {len(lbl_files)}")

        # 1. Check matching counts
        assert len(img_files) == len(lbl_files), f"❌ Image count ({len(img_files)}) does not match label count ({len(lbl_files)})!"

        # 2. Check each image has a label
        for img in img_files:
            lbl = lbl_dir / (img.stem + ".txt")
            assert lbl.exists(), f"❌ Label file missing for image: {img.name}"

            # 3. Check label contents and remapping
            with open(lbl, "r") as f:
                lines = f.readlines()
            
            for line_idx, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = int(parts[0])
                assert 0 <= class_id < total_count, f"❌ Out-of-bounds class ID {class_id} in {lbl.name} line {line_idx}"

                box_counter[class_id] += 1

                # Specific remapping validations
                if img.name.startswith("pieds_"):
                    assert class_id >= dino_count, f"❌ PIEDS image {img.name} contains invalid class ID {class_id} (< {dino_count})"
                else:
                    # It's a DINO image, prefix is the class name
                    dino_class = next((c for c in dino_classes if img.name.startswith(f"{c}_")), None)
                    assert dino_class is not None, f"❌ Could not determine DINO class from filename: {img.name}"
                    expected_idx = dino_classes.index(dino_class)
                    assert class_id == expected_idx, f"❌ DINO image {img.name} contains class ID {class_id}, expected {expected_idx} ({dino_class})"

    print("\n🎉 Integrity check PASSED successfully! No anomalies found.")
    
    print("\n📦 Bounding Box Class Distribution:")
    for idx, name in enumerate(combined_classes):
        cnt = box_counter[idx]
        print(f"  Class {idx:02d} ({name:<20}): {cnt} bounding boxes")

if __name__ == "__main__":
    main()
