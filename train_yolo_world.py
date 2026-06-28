import os
from ultralytics import YOLOWorld

def main():
    print("--- Starting fine-tuning of YOLO-World ---")

    # Path to dataset configuration
    data_yaml = "yolo_final/dataset.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ Error: Dataset config {data_yaml} not found. Please run merge_datasets_correct.py first.")
        return

    # Load YOLO-World v2 model
    # Note: yolov8s-worldv2.pt is version 2, which has improved architecture and accuracy
    model_name = "yolov8s-worldv2.pt"
    print(f"Loading base YOLO-World model: {model_name}")
    model = YOLOWorld(model_name)

    # Start training
    print("Initiating training loop...")
    model.train(
        data=data_yaml,
        epochs=50,             # Set to 50 epochs (modify as needed)
        imgsz=640,            # Standard image resolution
        batch=8,              # Reduced batch size to fit within 15GB unified memory (reduce to 4 if OOM occurs)
        device="mps",         # GPU acceleration on Mac (Apple Silicon)
        project="Unified_Vision",
        name="yolo_world_merged",
        save=True,            # Save training checkpoints and weights
        workers=4,            # Number of CPU worker threads for data loading
        patience=15           # Early stopping patience
    )
    print("Training process triggered successfully!")

if __name__ == "__main__":
    main()
