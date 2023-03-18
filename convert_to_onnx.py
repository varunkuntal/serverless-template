import torch
import onnx
import onnxoptimizer as optimizer
from onnxsim import simplify
from pytorch_model import Classifier

# Define the input and output names for the model
input_name = 'input'
output_name = 'output'

# Load the PyTorch model and its weights
model_path = 'model/pytorch_model_weights.pth'
model_dict = torch.load(model_path)
model = Classifier()
model.load_state_dict(model_dict)
model.eval()

# Create an example input tensor for the model
dummy_input = torch.randn(1, 3, 224, 224)

# Convert the PyTorch model to ONNX format
torch.onnx.export(model, dummy_input, 'model/base_model.onnx', verbose=True, input_names=[input_name],
                  output_names=[output_name], opset_version=11)

# Load the ONNX model and optimize it
onnx_model = onnx.load('model/base_model.onnx')
passes = ['extract_constant_to_initializer', 'eliminate_unused_initializer']
optimized_model = optimizer.optimize(onnx_model, passes)

# Simplify the optimized ONNX model
simplified_model, _ = simplify(optimized_model)

# Save the final ONNX model
onnx.save(simplified_model, 'model/model_optimized.onnx')
