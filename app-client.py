import streamlit as st
from PIL import Image
import requests
import numpy as np
import json
from app_utils import base64_encode_image, base64_decode_image
import asyncio
import aiohttp

st.set_page_config(
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
)

async def make_requests(data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url='http://127.0.0.1:8000/predict', data=data) as resp: #, timeout=aiohttp.ClientTimeout(total=5)
            response = await resp.json()
            return response

def image_to_array(image):
    file_name = image.name
    
    image = Image.open(image)
    im_size = image.size
    im_mode = image.mode

    if image.mode != "RGB":
        image = image.convert("RGB")

    np_image = np.array(image, 'float32')
    np_image_shape = np_image.shape
    np_image = np_image.copy(order="C")
    np_image = base64_encode_image(np_image)

    return file_name, im_size, im_mode, np_image, np_image_shape

def postprocess_predictions(response, np_image_shape):
    image_w_bbox = base64_decode_image(response['image_w_bbox'], 'float32', np_image_shape)
    image_w_bbox = Image.fromarray(np.uint8(image_w_bbox), mode='RGB')
    label, probability = response['label'], response['probability']
    return image_w_bbox, label, probability

async def get_all_artifacts(image):
    file_name, im_size, im_mode, np_image, np_image_shape = image_to_array(image)

    data = {
        'success': False,
        'image': np_image,
        'orig_shape': list(np_image_shape)
    }
    response = await make_requests(json.dumps(data))

    if response['success'] == True:
        image_w_bbox, label, probability = postprocess_predictions(response, np_image_shape)
        return file_name, im_size, im_mode, np_image, np_image_shape, response, image_w_bbox, label, probability

st.title('YOLOv4 - Object Detection')
st.sidebar.header('Options')

images = st.sidebar.file_uploader("Choose Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
# Add another set of images from list of urls separated by '|'

async def main(images):
    tasks = [asyncio.create_task(get_all_artifacts(image)) for image in images]
    task_results = await asyncio.gather(*tasks)
    return task_results

originals, modified = st.beta_columns(2)

if images is not None:
    task_results = asyncio.run(main(images))
    for task_result, (num, image) in zip(task_results, enumerate(images)):
        file_name, im_size, im_mode, np_image, np_image_shape, response, image_w_bbox, label, probability = task_result

        if response['success'] == True:
            originals.text("Input Image:")
            originals.text("Image #{}, Filename: {}, mode: {}, size: {}".format(num, file_name, im_mode, im_size) + " "*200 + "|")

            modified.text("Objects in the image:")
            modified.text(", ".join(["{} -- {}".format(l, p) for l, p in zip(label, probability)])+ " "*200 + "|")
            
            originals.image(image)
            modified.image(image_w_bbox)
    print(f"Processed {len(images)} images successfully!")