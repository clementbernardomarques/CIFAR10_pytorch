import torch
import onnx
import os
from models import cnn

# File with the best model weight
path_weight = "./checkpoint/lr_0.01_batch_32/ckpt.pth"

if not os.path.isdir('./best_model/'):
    os.mkdir('./best_model/')

# Initialization of the architecture of the model
model_to_export = cnn.Net()
model_to_export.load_state_dict(torch.load(path_weight, map_location=torch.device('cpu'))['net'])

# Preparation of dummy input
x = torch.randn(1, 3, 32, 32, requires_grad=True)
torch_out = model_to_export(x)

# Export the model
torch.onnx.export(model_to_export,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "./best_model/model_CIFAR10.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

# Load the ONNX model
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
