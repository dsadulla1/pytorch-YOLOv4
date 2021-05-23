import torch
import torch.autograd.profiler as profiler
from models import Yolov4
import numpy as np
import cv2

use_cuda = True
BATCH_SIZE = 5
MODEL_IMAGE_WIDTH = 768
MODEL_IMAGE_HEIGHT = 576
MODEL_IMAGE_CHANS = 3

def load_model(use_cuda):
    n_classes = 80
    weightfile = '../yolo-weights/yolov4.pth'
    device = 'cuda' if use_cuda else 'cpu'
    
    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    pretrained_dict = torch.load(weightfile, map_location=torch.device(device))
    model.load_state_dict(pretrained_dict)
    return model

model = load_model(use_cuda)
model.eval()

# inputs = torch.randint(low=0, high=255, size=(BATCH_SIZE, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANS), device='cuda' if use_cuda else 'cpu')
# inputs = torch.randint(low=0, high=255, size=(2,3,4,3), device='cuda' if use_cuda else 'cpu')

# if len(inputs.shape) == 3:
#     inputs = inputs.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
# elif len(inputs.shape) == 4:
#     inputs = inputs.permute(0, 3, 1, 2).float().div(255.0)

imgfile = './data/dog.jpg'
inputs = cv2.imread(imgfile)
inputs = np.vstack([inputs[np.newaxis]  for i in range(BATCH_SIZE)])

if len(inputs.shape) == 3:
    inputs = torch.from_numpy(inputs.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
elif len(inputs.shape) == 4:
    inputs = torch.from_numpy(inputs.transpose(0, 3, 1, 2)).float().div(255.0)

print(len(inputs.shape), inputs.shape)

out = model(inputs)
print('single run success!')

with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

with profiler.profile() as prof:
    with profiler.record_function("model_inference"):
        model(inputs)

prof.export_chrome_trace("trace.json")

# # errors out in torch=1.4.0; probably works in newer versions
# with profiler.profile(profile_memory=True, record_shapes=True) as prof:
#     model(inputs)

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
