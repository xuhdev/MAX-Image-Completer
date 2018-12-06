import pytest
import requests
import io
from PIL import Image


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'Model Asset Exchange Server'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'DCGAN'
    assert metadata['name'] == 'DCGAN Image Completer Model'
    assert metadata['description'] == 'DCGAN Image completion model; trained using Tensorflow on CelebA dataset'
    assert metadata['license'] == 'MIT'


def test_predict():
    model_endpoint = 'http://localhost:5000/model/predict?mask_type=left'
    file_path = 'assets/input/test_image.jpg'

    with open(file_path, 'rb') as file:
        file_form = {'file': (file_path, file, 'image/jpg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.content
    
    im = Image.open(io.BytesIO(response))
    #Final image is a collage of all results (19 intermediate + 1 final). Size of each image is 64 by 64.   
    #Combining all images give one output image of size 256 * 320
    assert im.size == (320, 256)
    #pixel values of output image which is the bottom right (20th) image
    pixel_face_1 = im.getpixel((270, 225))  
    pixel_face_2 = im.getpixel((280, 235)) 
    #pixels values should not be dark
    assert pixel_face_1[0] > 100
    assert pixel_face_1[1] > 50
    assert pixel_face_1[2] > 20
    
    assert pixel_face_2[0] > 120
    assert pixel_face_2[1] > 60
    assert pixel_face_2[2] > 30
    
    


if __name__ == '__main__':
    pytest.main([__file__])
