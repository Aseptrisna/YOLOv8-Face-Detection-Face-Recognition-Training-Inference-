
from ultralytics import YOLO
import cv2

model = YOLO("runs/classify/train/weights/best.pt")

img = cv2.imread("test.jpg")
result = model(img)[0]

pred = result.probs.top1
label = result.names[pred]
confidence = result.probs.top1conf

print("Nama:", label)
print("Confidence:", confidence)
