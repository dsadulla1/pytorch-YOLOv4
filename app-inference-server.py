
from torch import nn
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
from models import Yolov4
import torch
import cv2

from threading import Thread
import numpy as np
import base64
import redis
import time
import json
import sys

from app_utils import base64_encode_image, base64_decode_image

MODEL_IMAGE_WIDTH = 768
MODEL_IMAGE_HEIGHT = 576
MODEL_IMAGE_CHANS = 3

IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 5
SERVER_SLEEP = 0.25
use_cuda = True

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
    return model

def get_predictions(model, imgs, sized_imgs, use_cuda, verbose=0):
    if verbose:
        print("========","Start of get_predictions")
        print("imgs.shape: ",imgs.shape)
        print("sized_imgs.shape: ",sized_imgs.shape)
        print("use_cuda.value: ",use_cuda)

    boxes = do_detect(model, sized_imgs, 0.4, 0.6, use_cuda)
    if verbose:
        print("boxes.length", len(boxes))
        print("boxes", boxes)

    class_names = load_class_names('data/coco.names')
    results = plot_boxes_cv2(imgs, boxes[0], None, class_names) # check if this is batched or single

    if verbose:
        print("results[0]", results[0].shape)
        print("results[1]", results[1])
        print("results[2]", results[2])
        print("========","End of get_predictions")
    return results

def object_detection_process(use_cuda):
    print("* Loading model...")
    model = load_model(use_cuda)
    print("* Model loaded")

    while True:
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        print("Current length of queue:", len(queue), " and Batchsize is set to:", BATCH_SIZE)
        imageIDs = []
        img_batch = None
        sized_batch = None

        for q in queue: # imageID, image
            q = json.loads(q.decode("utf-8"))
            # image = base64_decode_image(q['payload']["image"], 'float32', tuple(q['payload']['orig_shape'])) # serialized np_image --> deserialized np_image
            image = base64_decode_image(q["image"], 'float32', tuple(q['orig_shape'])) # serialized np_image --> deserialized np_image
            sized_image = resize_image_for_model(image, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT)

            if img_batch is None:
                img_batch = image
                sized_batch = sized_image
            else:
                img_batch = np.vstack([img_batch, image])
                sized_batch = np.vstack([sized_batch, sized_image])

            imageIDs.append(q["id"])

        if len(imageIDs) > 0:
            print("* Batch size: {}".format(img_batch.shape))
            results = get_predictions(model, img_batch, sized_batch, use_cuda)

            for (imageID, resultSet) in zip(imageIDs, [results]):
                print("len of resultSet:", len(resultSet))
                img_w_bbox, label, prob = resultSet
                response_body = {
                    "image_w_bbox": base64_encode_image(img_w_bbox), 
                    "probability": [str(round(p,3)) for p in prob],
                    "label": label
                }
                db.set(imageID, json.dumps(response_body))
                print('processed imageID:', imageID)
            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)
        time.sleep(SERVER_SLEEP)

if __name__ == "__main__":
    print("* Starting model service...")
    object_detection_process(use_cuda=False)

    # t = Thread(target=object_detection_process, args=(True,))
    # t.daemon = True
    # t.start()