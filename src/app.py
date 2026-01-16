from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os
from ultralytics import YOLOWorld, YOLO
import torch
import yaml
import sys
from nob import Nob
import gc

MODELNAME = os.getenv("MODELNAME","yolov8x-worldv2")
FORMAT = os.getenv("FORMAT","ultralytics")
DEVICE = os.getenv("DEVICE","cpu")
HALF = os.getenv("HALF","False")
WORLD = os.getenv("WORLD","True")

def check_cuda():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA avaliable: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        x = torch.rand(5, 3).cuda()
        print("Tensor-operation on GPU successful.")
        return True
    else:
        return False
try:    
    if check_cuda() and DEVICE!="cpu": 
        device=DEVICE 
        #print(torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor) 
        freemem = torch.cuda.mem_get_info()[0]
        devicename=torch.cuda.get_device_name(device)
        print(f"starting on {devicename}, {freemem/1024/1024:.0f} MB left")
    else:
        device="cpu"
        print(f"starting on CPU")
except Exception as e:
    device='cpu'
    print(e)

app = Flask(__name__)
if WORLD=="True" or WORLD=="true" or WORLD=="1" or WORLD=="yes":
    model = YOLOWorld(f'{MODELNAME}.pt')
    model.to(device)
else:
    model = YOLO(f'{MODELNAME}.pt')
    model.to(device)

if HALF=="True" or HALF=="true" or HALF=="1" or HALF=="yes":
    half=True
else:
    half=False

if WORLD=="True" or WORLD=="true" or WORLD=="1" or WORLD=="yes":
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

if FORMAT == "engine":
    if os.path.exists(f"{MODELNAME}.engine"):
        print(f"Loading Engine: {MODELNAME}.engine")
        model = YOLO(f"{MODELNAME}.engine")
    else:
        export_path = model.export(
            format="engine", 
            device=device, 
            half=half,       
            simplify=True, 
            workspace=4 
        )
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = YOLO(export_path)

if (FORMAT == "onnx"):
    if os.path.exists(f"{MODELNAME}.onnx"):
        print(f"Loading Engine: {MODELNAME}.onnx")
        model = YOLO(f"{MODELNAME}.onnx")
    else:
        model.export(format="onnx")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model=YOLO(f"{MODELNAME}.onnx")

if(FORMAT=="openvino"):
    model.export(format="openvino")
    model=YOLO(f"{MODELNAME}_openvino_model/")

@app.route("/classes", methods=['POST','GET'])
def set_classes():
    if request.method=="GET":
        return {"classes": tuple(model.names)}
    classes=[]
    if len(request.data)>1:
        request_data = request.json
        classes=tuple(request_data.get("classes",[]))
    if len(classes)<1:
        return {"classes": tuple(model.names)}
    model.set_classes(classes)
    print(f"classes: {model.names}")
    return {"classes": classes}

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['image']        
    image=Image.open(file)

    
    results = model.predict(image, save=False, conf=0.25, half=half, device=device)  
    #results[0].save("debug.jpg")
    
    predictions=[]
    
    classes=results[0].boxes.cls.cpu().numpy().astype(int).tolist()
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
    
    #print(predictions)       
    return {"predictions": predictions}

if __name__ == "__main__":
    from waitress import serve
    
    serve(app, host="0.0.0.0", port=8040)
    
