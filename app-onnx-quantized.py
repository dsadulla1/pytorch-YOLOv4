
import numpy as np
from time import perf_counter
import onnxruntime

def timer(f,*args):   
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))

batch_size = 5
# x = torch.randint(0, 255, (batch_size, 3, 512, 512), requires_grad=False).float().div(255.)
x = np.random.randint(0, 255, (batch_size, 3, 512, 512)).astype(np.float32) / 255.
print(x.shape, x.min(), x.max())

print("-------------------- Onnx runtime Dynamic Quantized --------------------")

sess_options = onnxruntime.SessionOptions()

# sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 1

# model_path = "../yolo-weights/dynamic_quantized_model.onnx"
model_path = "../yolo-weights/model.onnx"

ort_session = onnxruntime.InferenceSession(model_path, sess_options)

print('onnxruntime.get_device: ', onnxruntime.get_device())

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: x} # to_numpy(x)
ort_outs = ort_session.run(None, ort_inputs)

def batch_test(ort_session, x):
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x} # to_numpy(x)
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

print("time-taken: ", np.mean([timer(batch_test, ort_session, x) for _ in range(10)]))