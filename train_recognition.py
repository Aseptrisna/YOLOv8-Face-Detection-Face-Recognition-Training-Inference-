from ultralytics import YOLO

# Load YOLOv8 Classification
model = YOLO("yolov8n-cls.pt")

# Train
model.train(
    data="datasets",
    epochs=30,
    imgsz=224,
)
