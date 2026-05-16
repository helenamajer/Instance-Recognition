from ultralytics import YOLO
from pathlib import Path

# Train YOLOv8 for instance recognition with prepared dataset.

DATASET = Path("data/yolo_dataset/dataset.yaml")
OUTPUT  = Path("models")

# Load the pretrained YOLOv8 segmentation model.
model = YOLO("yolov8n-seg.pt")

# Training parameters.
model.train(
    data=str(DATASET),
    # number of times the model cycles through each image in the dataset.
    # start with 50, increase if needed.
    epochs=50,
    # match the render resolution of raw dataset.
    imgsz=720,
    # Use batch=-1 to tigger AutoBatch.
    # Or set batch size (e.g. 16, 32, 64)
    batch=-1,
    # use Apply Silicon GPU.
    device="mps",
    project=str(OUTPUT),
    name="yolov8_parts",
    exist_ok=True,
)

print("Training complete.")
print(f"Weights saved to: models/yolov8_parts/weights/best.pt")