import os
import torch
from ultralytics import YOLO

os.chdir('/home/Documents/Analytics/Computer Vision/CB_Analytics_WS/GIS_WS/GIS_Roads_WS/YOLOV8 Dataset')
home_direc = os.getcwd()

torch.cuda.empty_cache()

yolo_dataset_direc = os.path.join(home_direc, 'data.yaml')

model = YOLO('yolov8x-seg.yaml')
model.train(data=yolo_dataset_direc, epochs=10, imgsz=640, batch=2, device='cpu', workers=0, amp=True)
