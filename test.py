import numpy as np
import requests
from PIL import Image

response = requests.post(
                "http://localhost:5000/predict",
                data = {"api_key": "abc"},
                files={"image": open("bus.jpg", 'rb')},
                timeout=10,
            )
print (response.json())
