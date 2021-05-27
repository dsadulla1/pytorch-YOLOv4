from PIL import Image
import numpy as np
import json
from app_utils import base64_encode_image, base64_decode_image
import asyncio
import aiohttp
import time
import argparse

async def make_requests(data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url='http://127.0.0.1:8000/predict', data=data) as resp: #, timeout=aiohttp.ClientTimeout(total=5)
            response = await resp.json()
            return response

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

def postprocess_predictions(response, np_image_shape):
    image_w_bbox = base64_decode_image(response['image_w_bbox'], 'float32', np_image_shape)
    image_w_bbox = Image.fromarray(np.uint8(image_w_bbox), mode='RGB')
    label, probability = response['label'], response['probability']
    return image_w_bbox, label, probability

async def get_all_artifacts(image_path, id):
    start = time.perf_counter()
    im_size, im_mode, np_image, np_image_shape = read_convert_image_to_array(image_path)

    data = {
        'success': False,
        'image': np_image,
        'orig_shape': list(np_image_shape)
    }
    response = await make_requests(json.dumps(data))

    if response['success'] == True:
        image_w_bbox, label, probability = postprocess_predictions(response, np_image_shape)
        end = time.perf_counter()
        return {'id': id, 'start': start, 'end': end}
    else:
        None

async def main(test_image_path, n_reqs, async_mode):
    tasks = [ asyncio.create_task(get_all_artifacts(test_image_path, i)) for i in range(n_reqs) ]
    if async_mode == 'gather':
        task_results = await asyncio.gather(*tasks)
    if async_mode == 'as_completed': 
        task_results = []
        for t in asyncio.as_completed(tuple(tasks)):
            t2 = await t
            task_results.append(t2)
    return task_results

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_reqs", type=int, default=20, help="How many asynchronous requests?")
    parser.add_argument("-t", "--tag", type=str, default='No tag', help="Give this experiment a name or tag?")
    parser.add_argument("-a", "--async_mode", choices=['as_completed', 'gather'], type=str, default='as_completed', help="Choose which async method to use.")
    args = parser.parse_args()

    n_reqs = args.n_reqs
    tag = args.tag
    async_mode = args.async_mode

    start = time.perf_counter()
    test_image_path = './data/dog.jpg'
    task_results = asyncio.run(main(test_image_path, n_reqs, async_mode))
    end = time.perf_counter()

    print("Calculating...")
    def duration_(di): 
        return di['end'] - di['start']
    n_unsuccess = len([i for i in task_results if i is None])
    n_success = len([i for i in task_results if i is not None])
    total_time_successes = sum([duration_(i) for i in task_results if i is not None])

    print('='*25, f"Summary for {tag} working {async_mode}", '='*25)
    print("Number of requests sent:", n_reqs)
    print("Number of unsuccessful requests:", n_unsuccess)
    print("Number of successful requests:", n_success)
    print("Average response time per requests (successes only):", total_time_successes / n_success)
    print("Average response time per requests (both):", total_time_successes / n_reqs)
    print("Total time taken:", end - start, 'seconds')

# python app-api-performance.py -n 100 -t "CPU, Batch=5" -a "as_completed" >> app-perf-logs/20210525.log
# python app-api-performance.py -n 100 -t "CPU, Batch=5" -a "gather" >> app-perf-logs/20210525.log
# python app-api-performance.py -n 100 -t "GPU, Batch=5" -a "as_completed" >> app-perf-logs/20210525.log
# python app-api-performance.py -n 100 -t "GPU, Batch=5" -a "gather" >> app-perf-logs/20210525.log