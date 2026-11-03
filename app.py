from flask import Flask, render_template, request, redirect, url_for
import json
import io
from PIL import Image
import time
import threading
import os
import shutil
import tempfile
from ultralytics import YOLOWorld, YOLOE, YOLO
import torch
import yaml
from nob import Nob

MODELNAME = os.getenv("MODELNAME","yolov8x-worldv2")
# "engine" is the spelling the upstream latest image used; accept both instead of silently running the plain .pt
FORMAT = os.getenv("FORMAT","ultralytics").lower()
FORMAT = "tensorrt" if FORMAT == "engine" else FORMAT
if FORMAT not in ("ultralytics", "onnx", "openvino", "tensorrt"):
    raise SystemExit(f"FORMAT must be ultralytics, onnx, openvino or tensorrt (got '{os.getenv('FORMAT')}')")
# QUANTIZE is the single precision knob: 8, 16 or nothing (=32); int8/fp16/fp32 are accepted aliases
QUANTIZE = {"8": 8, "int8": 8, "16": 16, "fp16": 16, "32": 32, "fp32": 32}.get(os.getenv("QUANTIZE", "32").lower())
if QUANTIZE is None:
    raise SystemExit(f"QUANTIZE must be 8, 16 or 32 (got '{os.getenv('QUANTIZE')}')")
if QUANTIZE == 8 and FORMAT == "ultralytics":
    raise SystemExit("INT8 only exists in exported artifacts - set FORMAT=tensorrt (or onnx/openvino); "
                     "the plain .pt cannot run INT8")
PORT = int(os.getenv("PORT", 4000))
CALIB_DIR = os.getenv("CALIB_DIR", "calib")       # images/ here is used for TensorRT INT8 calibration
# predict itself understands quantize=16 only; INT8 is baked into the exported artifact
PREDICT_KWARGS = {} if FORMAT in ("onnx", "openvino", "tensorrt") else ({"quantize": 16} if QUANTIZE == 16 else {})

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__)
# YOLOE needs its own wrapper: it computes the text embeddings set_classes() requires
model = (YOLOE if MODELNAME.startswith("yoloe") else YOLOWorld)(f'{MODELNAME}.pt')

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

def _quantize_arg():
    """ultralytics replaced the deprecated half=/int8= export kwargs with one quantize= scheme."""
    return {} if QUANTIZE == 32 else {"quantize": QUANTIZE}

def _calib_data():
    """TensorRT INT8 calibrates on ultralytics' data= dataset, which defaults to coco8 (4 images).
    Point it at our own camera snapshots; see fetch_calibration_images.py."""
    images = os.path.join(CALIB_DIR, "images")
    n = len([f for f in os.listdir(images)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]) if os.path.isdir(images) else 0
    if n < 1:
        print(f"WARNING: no calibration images in {images} - TensorRT will calibrate on coco8's 4 images. "
              f"Fill it with ~300 frigate snapshots (fetch_calibration_images.py).")
        return None
    print(f"calibration: {n} images from {images}" + ("" if n >= 300 else " - <300 recommended for INT8"))
    # CALIB_DIR is mounted read-only, so the dataset yaml goes to a writable temp dir; its path: is absolute
    data_yaml = os.path.join(tempfile.gettempdir(), "yolostack_calib.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(images)}\ntrain: .\nval: .\nnc: 1\nnames: [object]\n")
    return data_yaml

if FORMAT == "onnx":
    onnx_export_path = f"{MODELNAME}.onnx"
    final_path = os.path.join(MODELS_DIR, f"{MODELNAME}_onnx_{QUANTIZE}.onnx")
    if os.path.exists(final_path):
        print(f"Reusing cached model: {final_path}")
    else:
        model.export(format="onnx", **({"quantize": 16} if QUANTIZE == 16 else {}))
        if QUANTIZE == 8:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(onnx_export_path, final_path, weight_type=QuantType.QUInt8)
            os.remove(onnx_export_path)
            print(f"ONNX model dynamically quantized to INT8: {final_path}")
        else:
            shutil.move(onnx_export_path, final_path)
    del model
    _free_model()
    model = YOLO(final_path)

elif FORMAT == "openvino":
    final_path = os.path.join(MODELS_DIR, f"{MODELNAME}_openvino_{QUANTIZE}")
    if os.path.exists(final_path):
        print(f"Reusing cached model: {final_path}")
    else:
        model.export(format="openvino", **_quantize_arg())
        shutil.move(f"{MODELNAME}_openvino_model", final_path)
    del model
    _free_model()
    model = YOLO(final_path)

elif FORMAT == "tensorrt":
    final_path = os.path.join(MODELS_DIR, f"{MODELNAME}_tensorrt_{QUANTIZE}.engine")
    if os.path.exists(final_path):
        print(f"Reusing cached model: {final_path}")
    else:
        model.export(format="engine", **_quantize_arg(), **({"data": _calib_data()} if QUANTIZE == 8 else {}))
        shutil.move(f"{MODELNAME}.engine", final_path)
    del model
    _free_model()
    model = YOLO(final_path)

try:
    devcount=torch.cuda.device_count()
    if devcount>0:   
        device='cuda:0'
        model.to(device)
        freemem = torch.cuda.mem_get_info()[0]
        devicename=torch.cuda.get_device_name(device)
        print(f"running on {devicename}, {freemem/1024/1024:.0f} MB left")
        print(f"format={FORMAT}, quantize={QUANTIZE}")
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
        results = model.predict(image, save=False, conf=0.25, device=device, **PREDICT_KWARGS)
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
    
