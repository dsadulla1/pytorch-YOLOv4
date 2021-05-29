from models import Yolov4
import torch
import numpy as np
from time import perf_counter
from torch import nn

def timer(f,*args):   
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))

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

batch_size = 5
x = torch.randint(0, 255, (batch_size, 3, 512, 512), requires_grad=False).float().div(255.)

print('-------------------- CPU Performance --------------------')

print("-------------------- Baseline performance --------------------")
model = load_model(use_cuda=False)
model.eval()
print("time-taken: ", np.mean([timer(model,x) for _ in range(10)]))
ans = model(x)
if ans: print("loaded properly")

print("-------------------- Dynamic Quantized performance --------------------")
backend = "fbgemm"
dynamic_quantized_model = torch.quantization.quantize_dynamic(model, {nn.Conv2d}, dtype=torch.qint8) # , nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU, nn.MaxPool2d
print("time-taken: ", np.mean([timer(dynamic_quantized_model, x) for _ in range(10)]))
torch.save(dynamic_quantized_model,'../yolo-weights/dynamic_quantized_model.pt')

loaded_dynamic_quantized_model = torch.load('../yolo-weights/dynamic_quantized_model.pt')
ans = loaded_dynamic_quantized_model(x)
if ans: print("loaded properly")


print("-------------------- Onnx runtime Baseline --------------------")
import onnx
import onnxruntime

torch_out = model(x)

# Export the model
torch.onnx.export(model,                                        # model being run
                  x,                                            # model input (or a tuple for multiple inputs)
                  "../yolo-weights/model.onnx",                 # where to save the model (can be a file or file-like object)
                  export_params=True,                           # store the trained parameter weights inside the model file
                  opset_version=13,                             # the ONNX version to export the model to
                  do_constant_folding=True,                     # whether to execute constant folding for optimization
                  input_names = ['input'],                      # the model's input names
                  output_names = ['output'],                    # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},   # variable length axes
                                "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}})

onnxmodel = onnx.load("../yolo-weights/model.onnx")
onnx.checker.check_model(onnxmodel)

ort_session = onnxruntime.InferenceSession("../yolo-weights/model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-01, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def batch_test(ort_session, x):
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

print("time-taken: ", np.mean([timer(batch_test, ort_session, x) for _ in range(10)]))

print("-------------------- Onnx runtime Dynamic Quantized --------------------")
import onnx 
import onnxruntime

sess_options = onnxruntime.SessionOptions()

sess_options.intra_op_num_threads = 2
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

torch_out = loaded_dynamic_quantized_model(x)

# Export the model
torch.onnx.export(loaded_dynamic_quantized_model,                   # model being run
                  x,                                                # model input (or a tuple for multiple inputs)
                  "../yolo-weights/dynamic_quantized_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,                               # store the trained parameter weights inside the model file
                  opset_version=13,                                 # the ONNX version to export the model to
                  do_constant_folding=True,                         # whether to execute constant folding for optimization
                  input_names = ['input'],                          # the model's input names
                  output_names = ['output'],                        # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}})

dynamic_quantized_onnxmodel = onnx.load("../yolo-weights/dynamic_quantized_model.onnx")
onnx.checker.check_model(dynamic_quantized_onnxmodel)

ort_session = onnxruntime.InferenceSession("../yolo-weights/dynamic_quantized_model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-01, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def batch_test(ort_session, x):
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

print("time-taken: ", np.mean([timer(batch_test, ort_session, x) for _ in range(10)]))


# print("-------------------- Scripted performance --------------------")
# scripted_model = torch.jit.script(model)
# torch.jit.save(scripted_model,'../yolo-weights/scripted_model.pt')
# print("time-taken: ", np.mean([timer(scripted_model,x) for _ in range(10)]))

# loaded_scripted_model = torch.jit.load('../yolo-weights/scripted_model.pt')
# ans = loaded_scripted_model(x)
# if ans: print("loaded properly")


# print("-------------------- Dynamic Quantized & Scripted performance --------------------")
# dynamic_quantized_scripted_model = torch.jit.script(dynamic_quantized_model, x)
# torch.jit.save(dynamic_quantized_scripted_model,'../yolo-weights/dynamic_quantized_model.pt')
# print("time-taken: ", np.mean([timer(dynamic_quantized_scripted_model, x) for _ in range(10)]))

# loaded_dynamic_quantized_scripted_model = torch.jit.load('../yolo-weights/dynamic_quantized_scripted_model.pt')
# ans = loaded_dynamic_quantized_scripted_model(x)
# print("loaded properly")


######## References
# https://spell.ml/blog/pytorch-quantization-X8e7wBAAACIAHPhT