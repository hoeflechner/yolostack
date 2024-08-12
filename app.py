from flask import Flask, render_template, request, redirect, url_for
import json
import io
from PIL import Image
import time
import threading
import os
from ultralytics import YOLOWorld, YOLO
import torch
import yaml
from nob import Nob

MODELNAME = os.getenv("MODELNAME","yolov8x-worldv2")
FORMAT = os.getenv("FORMAT","ultralytics")
half=True

app = Flask(__name__)
model = YOLOWorld(f'{MODELNAME}.pt')

device='cpu'

labels=set([])
with open("config.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
    nobTree=Nob(data_loaded)
    d=[]
    for track in nobTree.find("track"):
        d+=nobTree[track]
    for i in d:
        labels.add(i.val)
    print(f"labels: {labels}")
model.set_classes(list(labels))

if (FORMAT=="onnx"):
    model.export(format="onnx") 
    model=YOLO(f"{MODELNAME}.onnx")
    half=False

if(FORMAT=="openvino"):
    model.export(format="openvino")
    model=YOLO(f"{MODELNAME}_openvino_model/")
    half=False

try:
    devcount=torch.cuda.device_count()
    if devcount>0:   
        device='cuda:0'
        model.to(device)
        freemem = torch.cuda.mem_get_info()[0]
        devicename=torch.cuda.get_device_name(device)
        print(f"running on {devicename}, {freemem/1024/1024:.0f} MB left")
except Exception as e:
    device='cpu'
    print(e)

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['image']
    image=Image.open(file)
    results = model.predict(image, save=False, conf=0.25, half=half, device=device)

    #results[0].save("debug.jpg")
    
    predictions=[]
    
    classes=results[0].boxes.cls.cpu().numpy().tolist()
    xyxy = results[0].boxes.xyxy.cpu().numpy().tolist()
    confidence = results[0].boxes.conf.cpu().numpy().tolist()
    
    for i in range(len(results[0].boxes)):
        box={}
        box['label']=results[0].names[classes[i]]
        box['confidence']=confidence[i]
        box['x_min']=int(xyxy[i][0])
        box['y_min']=int(xyxy[i][1])
        box['x_max']=int(xyxy[i][2])
        box['y_max']=int(xyxy[i][3])
        predictions.append(box)
    
    print(predictions)       
    return {"predictions": predictions}

if __name__ == "__main__":
    from waitress import serve
    
    serve(app, host="0.0.0.0", port=8040)
    
