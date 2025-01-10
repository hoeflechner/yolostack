import numpy as np
import requests
import json
from PIL import Image

def classes(classes=[]):
    if len(classes)<1:
        response = requests.get("http://localhost:8040/classes")
        return response
    response = requests.post(
                    "http://localhost:8040/classes",
                    json  = {"classes": classes}
                )
    return response


def predict(image, classes=[]):
    if len(classes)<1:
        response = requests.post(
                        "http://localhost:8040/predict",
                        data = {"api_key": "abc"},
                        files={"image": open(image, 'rb')},
                        timeout=20
                    )
        return response
    response = requests.post(
                        "http://localhost:8040/predict",
                        json = {"classes": classes},
                        files={"image": open(image, 'rb')},
                        timeout=20
                    )
    return response

print(json.dumps(classes(["sneaker","man"]).json()))
print(json.dumps(predict("bus.jpg").json(), indent=2))
print(json.dumps(classes(["red bus","blue bus"]).json()))
print(json.dumps(predict("bus.jpg").json(), indent=2))
print(json.dumps(classes().json()))