import cv2
from ultralytics import YOLO
import torch
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# torch.cuda.set_device(0)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# TRAINING

if __name__ == '__main__':      
    results = model.train(data="config.yaml", epochs=50)