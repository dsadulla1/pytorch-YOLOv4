import numpy as np
from time import perf_counter
import onnxruntime


def timer(f,*args):   
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))


batch_size = 5
quantized_model_path = "../yolo-weights/model-quantized-by-onnx.onnx"
x = np.random.randint(0, 255, (batch_size, 3, 512, 512)).astype(np.float32) / 255.
print(x.shape, x.min(), x.max())

session_optimizations = False

if session_optimizations:
    sess_options = onnxruntime.SessionOptions()
    # sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(quantized_model_path, sess_options)
else:
    ort_session = onnxruntime.InferenceSession(quantized_model_path)

print('onnxruntime.get_device: ', onnxruntime.get_device())

def batch_test(ort_session, x):
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x} # to_numpy(x)
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

print("time-taken: ", np.mean([timer(batch_test, ort_session, x) for _ in range(10)]))