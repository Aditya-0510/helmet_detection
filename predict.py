from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")

# Webcam inference
results = model(source=0, show=True, conf=0.5)
