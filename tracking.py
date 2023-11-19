import cv2
from ultralytics import YOLO

# PRE-TRAINED MODEL
model = YOLO('runs/detect/train2/weights/best.pt')  

# INFERENCE
results = model.track(source="clips/08fd33_0.mp4", show=True, tracker="botsort.yaml")
