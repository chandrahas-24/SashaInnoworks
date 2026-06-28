from ultralytics import YOLO

print("--- Starting 50-Epoch Training Loop ---")

# Load base model
model = YOLO("yolo11s.pt")

# Train
model.train(
    data="pieds_15_class.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="mps",
    project="PIEDS_Vision",
    name="pieds_15_class",
    save=True,
    workers=4,
    patience=15
)