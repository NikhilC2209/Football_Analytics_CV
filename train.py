import cv2
from ultralytics import YOLO
import torch
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# torch.cuda.set_device(0)

# Load the YOLOv8 model
model = YOLO('YOLO_weights/yolov8n.pt')

# TRAINING

if __name__ == '__main__':      
    results = model.train(data="config.yaml", epochs=100, patience=10, imgsz=1088)
    #results = model.train(data="config.yaml", epochs=10)

# IMGSZ = 1088

## train  ---------> yolov8n.pt
## train2 ---------> yolov8s.pt

# DATASET v6 & IMGSZ = 1088

## train4 ---------> yolov8s.pt
    
# IMGSZ = 640

## train3 ---------> yolov8x.pt (Trained on Colab)
## train5 ---------> yolov8n.pt
## train6 ---------> yolov8s.pt