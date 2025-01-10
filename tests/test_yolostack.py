
from io import BytesIO

def test_classes(app, client):
    response = client.post('/classes', json={"classes": ["a", "b", "c"]})
    assert response.status_code == 200
    assert 'classes' in response.json
    assert 'a' in response.json['classes']
    assert 'b' in response.json['classes']
    assert 'c' in response.json['classes']

def test_pediction(app, client):
    with open('bus.jpg', 'rb') as img:
        imgbytes = BytesIO(img.read())
    
    client.post('/classes', json={"classes": ["blue bus", "red bus"]})
    response = client.post('/predict', data = {'image': (imgbytes, "bus.jpg")})
    assert response.status_code == 200
    assert 'predictions' in response.json
    predictions=response.json["predictions"]
    assert "blue bus" in response.json["predictions"][0]["label"]