import streamlit as st
from PIL import Image
import numpy as np
import base64
import sys
import json
import os

def base64_encode_image(a):
    return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    return a


def read_convert_image_to_array(image_path):
    
    image = Image.open(image_path)
    im_size = image.size
    im_mode = image.mode

    if image.mode != "RGB":
        image = image.convert("RGB")

    np_image = np.array(image, 'float32')
    np_image_shape = np_image.shape
    np_image = np_image.copy(order="C")
    np_image = base64_encode_image(np_image)

    return im_size, im_mode, np_image, np_image_shape


def make_jsons(image_path, save_json_path):
    *_, np_image, np_image_shape = read_convert_image_to_array(image_path)

    data = {
        'success': False,
        'image': np_image,
        'orig_shape': list(np_image_shape)
    }
    
    with open(save_json_path, 'w') as json_file:
        json.dump(data, json_file)


if __name__ == '__main__':
    
    if not os.path.exists('./data/json-files'): os.mkdir('./data/json-files')
    
    image_paths = [f'./data/{file_}' for file_ in os.listdir('../data') if file_.endswith('.jpg')]
    save_json_paths = [file_.split('.')[0] for file_ in os.listdir('../data') if file_.endswith('.jpg')]
    save_json_paths = [f'./data/json-files/{file_}.json' for file_ in save_json_paths]

    for in_, out_ in zip(image_paths, save_json_paths):
        make_jsons(in_, out_)