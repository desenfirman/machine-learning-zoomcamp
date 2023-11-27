from io import BytesIO
from urllib import request
import tflite_runtime.interpreter as tflite
import numpy as np

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def rescale_img_array(x):
    return x / 255


interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()

url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"

img = download_image(url)
prepared_img = prepare_image(img, (150, 150))
img_array = np.array(prepared_img, dtype='float32')
img_array = rescale_img_array(img_array)

imgs_array = np.array([img_array])

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(input_index, imgs_array)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)

print(f"The result of prediction of url {url} is {preds}")