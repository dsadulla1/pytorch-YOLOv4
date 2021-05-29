### Onnx quantization 

import onnx
from onnxruntime.quantization import QuantizationMode, quantize

onnx_model_path = "../yolo-weights/model.onnx"
quantized_model_path = "../yolo-weights/model-quantized-by-onnx.onnx"

onnx_model = onnx.load(onnx_model_path)

quantized_model = quantize(
    model=onnx_model,
    quantization_mode=QuantizationMode.IntegerOps,
    force_fusions=True,
    symmetric_weight=True,
)

onnx.save_model(quantized_model, quantized_model_path)