from PIL import Image
import numpy as np
import base64

from typing import List as typing_list
from fastapi import FastAPI
from pydantic import BaseModel

import uuid
import io
import json
import redis
import time
import json

from app_utils import base64_encode_image

IMAGE_QUEUE = "image_queue"
CLIENT_SLEEP = 0.25

app = FastAPI()
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None

class RequestIn(BaseModel):
    success: bool = False
    image: str
    orig_shape: typing_list[int]
from fastapi.encoders import jsonable_encoder
class RequestOut(BaseModel):
    success: bool
    image: str
    orig_shape: typing_list[int]
    id: str
    image_w_bbox: str
    label: str
    probability: str

@app.post("/predict") # , response_model=RequestOut
def predict(data: RequestIn):
# @app.post("/predict")
# def predict(data):
    # print(type(data))
    # print({k:type(v) for k,v in json.loads(data).items()})
    
    k = str(uuid.uuid4())

    data = jsonable_encoder(data)
    data['id'] = k
    db.rpush(IMAGE_QUEUE, json.dumps(data))


    # redis_input = {}
    # redis_input['id'] = k
    # redis_input['payload'] = data.dict()
    # redis_input = jsonable_encoder(redis_input)
    # db.rpush(IMAGE_QUEUE, json.dumps(redis_input))

    while True:
        output = db.get(k)
        if output is not None:
            output = output.decode("utf-8")
            output = json.loads(output) # {"image_w_bbox": img_w_bbox, "label": label, "probability": prob}
            data["image_w_bbox"] = output["image_w_bbox"]
            data["label"] = output["label"]
            data["probability"] = output["probability"]
            db.delete(k)
            break
        time.sleep(CLIENT_SLEEP)
    data["success"] = True
    # data = jsonable_encoder(data)
    return data
    # return json.dumps(data)

# uvicorn app-server:app --reload