from pathlib import Path
from collections import Counter

val_imgs = list(Path("/content/yolo_final/images/val").glob("*.jpg"))
train_imgs = list(Path("/content/yolo_final/images/train").glob("*.jpg"))

print(f"Val: {len(val_imgs)} images")
print(f"Train: {len(train_imgs)} images")

# Show first 20 val filenames
print("\nSample val files:")
for f in val_imgs[:20]:
    print(f.name)