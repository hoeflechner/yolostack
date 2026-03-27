from flask import Flask, render_template, request, redirect, url_for
import json
import io
from PIL import Image
import time
import threading
import os
import shutil
from ultralytics import YOLOWorld, YOLO
import torch
import yaml
from nob import Nob

MODELNAME = os.getenv("MODELNAME","yolov8x-worldv2")
FORMAT = os.getenv("FORMAT","ultralytics")       # ultralytics, onnx, openvino, tensorrt
QUANTIZE = os.getenv("QUANTIZE","fp16")           # fp32, fp16, int8
PORT = int(os.getenv("PORT", 4000))
half = (QUANTIZE == "fp16")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

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

def _free_model():
    """Free VRAM after export/reload."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

if FORMAT == "onnx":
    onnx_export_path = f"{MODELNAME}.onnx"
    final_path = os.path.join(MODELS_DIR, f"{MODELNAME}_onnx_{QUANTIZE}.onnx")
    if os.path.exists(final_path):
        print(f"Reusing cached model: {final_path}")
    else:
        model.export(format="onnx", half=(QUANTIZE == "fp16"))
        if QUANTIZE == "int8":
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(onnx_export_path, final_path, weight_type=QuantType.QUInt8)
            os.remove(onnx_export_path)
            print(f"ONNX model dynamically quantized to INT8: {final_path}")
        else:
            shutil.move(onnx_export_path, final_path)
    del model
    _free_model()
    model = YOLO(final_path)
    half = False

elif FORMAT == "openvino":
    final_path = os.path.join(MODELS_DIR, f"{MODELNAME}_openvino_{QUANTIZE}")
    if os.path.exists(final_path):
        print(f"Reusing cached model: {final_path}")
    else:
        model.export(format="openvino", int8=(QUANTIZE == "int8"), half=half)
        shutil.move(f"{MODELNAME}_openvino_model", final_path)
    del model
    _free_model()
    model = YOLO(final_path)
    half = False

elif FORMAT == "tensorrt":
    final_path = os.path.join(MODELS_DIR, f"{MODELNAME}_tensorrt_{QUANTIZE}.engine")
    if os.path.exists(final_path):
        print(f"Reusing cached model: {final_path}")
    else:
        model.export(format="engine", half=half, int8=(QUANTIZE == "int8"))
        shutil.move(f"{MODELNAME}.engine", final_path)
    del model
    _free_model()
    model = YOLO(final_path)
    half = False

try:
    devcount=torch.cuda.device_count()
    if devcount>0:   
        device='cuda:0'
        model.to(device)
        freemem = torch.cuda.mem_get_info()[0]
        devicename=torch.cuda.get_device_name(device)
        print(f"running on {devicename}, {freemem/1024/1024:.0f} MB left")
        print(f"format={FORMAT}, quantize={QUANTIZE}, half={half}")
except Exception as e:
    device='cpu'
    print(e)

default_labels = list(labels)
model_lock = threading.Lock()

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['image']
    image=Image.open(file)

    # Optional: custom classes per request (comma-separated or JSON array)
    custom_classes = request.form.get('classes') or request.args.get('classes')
    if custom_classes:
        try:
            req_classes = json.loads(custom_classes)
        except (json.JSONDecodeError, TypeError):
            req_classes = [c.strip() for c in custom_classes.split(',') if c.strip()]
    else:
        req_classes = None

    with model_lock:
        if req_classes:
            model.set_classes(req_classes)
        results = model.predict(image, save=False, conf=0.25, half=half, device=device)
        if req_classes:
            model.set_classes(default_labels)

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
    
    serve(app, host="0.0.0.0", port=PORT)
    
