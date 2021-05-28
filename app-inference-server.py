from torch import nn
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
from models import Yolov4
import torch
import cv2
import random

import numpy as np
import redis
import time
import json

import argparse
from concurrent.futures import ProcessPoolExecutor

from app_utils import base64_encode_image, base64_decode_image

MODEL_IMAGE_WIDTH = 512 # 768
MODEL_IMAGE_HEIGHT = 512 # 576
MODEL_IMAGE_CHANS = 3

IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 5
SERVER_SLEEP = 0.25

db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None

def resize_image_for_model(image, width, height):
    sized_image = cv2.resize(image, (width, height))
    sized_image = cv2.cvtColor(sized_image, cv2.COLOR_BGR2RGB)
    return sized_image

def load_model(use_cuda):
    n_classes = 80
    weightfile = '../yolo-weights/yolov4.pth'
    device = 'cuda' if use_cuda else 'cpu'
    
    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    pretrained_dict = torch.load(weightfile, map_location=torch.device(device))
    model.load_state_dict(pretrained_dict)
    if use_cuda:
        model.cuda()
    return model

def get_predictions(model, imgs, sized_imgs, use_cuda, verbose=0):
    if verbose:
        print("========","Start of get_predictions")
        print("imgs.shape: ",len(imgs))
        print("sized_imgs.shape: ",sized_imgs.shape)
        print("use_cuda.value: ",use_cuda)

    boxes = do_detect(model, sized_imgs, 0.4, 0.6, use_cuda)
    if verbose:
        print("boxes.length", len(boxes))
        print("boxes", boxes)

    class_names = load_class_names('data/coco.names')
    results = [plot_boxes_cv2(img, box[0], None, class_names) for img, box in zip(imgs, boxes)]

    if verbose:
        print([[type(e) for e in el] for el in results])
        print("========","End of get_predictions")
    return results

def object_detection_process(use_cuda, verbose=False, enable_locks=True):
    print("* Loading model...")
    model = load_model(use_cuda)
    print("* Model loaded")

    while True:
        if enable_locks:
            try: # https://github.com/andymccurdy/redis-py#locks-as-context-managers
                with db.lock('my-lock-key', blocking_timeout=SERVER_SLEEP) as locked:
                    queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
            except redis.LockError as LE:
                time.sleep(SERVER_SLEEP)
                with db.lock('my-lock-key', blocking_timeout=SERVER_SLEEP) as locked:
                    queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
            finally:
                if len(queue) > 0: 
                    db.ltrim(IMAGE_QUEUE, len(queue), -1)
                    if verbose: print("Current length of queue:", len(queue))
        else:
            queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
            if len(queue) > 0: 
                db.ltrim(IMAGE_QUEUE, len(queue), -1)
                if verbose: print("Current length of queue:", len(queue))
        
        imageIDs = []
        img_batch = None
        sized_batch = None

        for q in queue: # imageID, image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], 'float32', tuple(q['orig_shape']))
            sized_image = resize_image_for_model(image, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT)

            if img_batch is None:
                img_batch = [image]
                sized_batch = sized_image[np.newaxis]
            else:
                img_batch.append(image)
                sized_batch = np.vstack([sized_batch, sized_image[np.newaxis]])

            imageIDs.append(q["id"])

        if len(imageIDs) > 0:
            try:
                results = get_predictions(model, img_batch, sized_batch, use_cuda)
            except RuntimeError as e:
                random.shuffle(queue)
                for q in queue: db.rpush(IMAGE_QUEUE, q)
                print("Returning a batch of images back to the queue for processing it later due to error..")
            else:
                for (imageID, resultSet) in zip(imageIDs, results):
                    image_w_bbox, label, prob = tuple(resultSet)
                    response_body = {
                        "image_w_bbox": base64_encode_image(image_w_bbox), 
                        "probability": [str(round(p,3)) for p in prob],
                        "label": label
                    }
                    db.set(imageID, json.dumps(response_body))
                    print('processed imageID:', imageID)
        time.sleep(SERVER_SLEEP)

if __name__ == "__main__":
    print("Batch Size:", BATCH_SIZE)
    print("SERVER_SLEEP:", SERVER_SLEEP)
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--use_cuda", help="True if GPU else False", type=str, choices=["T", "F"], default="F")
    parser.add_argument("-l", "--enable_locks", help="If lock should be enable for accessing redis queue", type=str, choices=["T", "F"], default="F")
    parser.add_argument("-t", "--tag_process", help="Name/Tag this process to identify within logs", type=str, required=False)
    args = parser.parse_args()

    use_cuda = True if args.use_cuda == "T" else False
    enable_locks = True if args.enable_locks == "T" else False
    
    if args.tag_process is not None:
        tag_process = str(args.tag_process) 
        print(f"* Starting model service with tag: {tag_process} *")
    else: 
        print(f"* Starting model service *")
    
    object_detection_process(use_cuda=use_cuda, enable_locks=enable_locks)