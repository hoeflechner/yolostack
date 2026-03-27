import numpy as np
import requests
import json
from PIL import Image

response = requests.post(
                "http://localhost:4000/predict",
                data = {"api_key": "abc"},
                files={"image": open("bus.jpg", 'rb')},
                timeout=10,
            )
print (json.dumps(response.json(), indent=2))
