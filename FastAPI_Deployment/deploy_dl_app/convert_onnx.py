import torch
from model import MLP
from ptl_modules import BankNoteClassifier
import numpy as np

state_dict = torch.load('bank_model.ckpt')["state_dict"]


net = MLP(4, 2)
torch_model = BankNoteClassifier(net)
torch_model.load_state_dict(state_dict)

# print(model)

# model.eval()

batch_size = 1

# Input to the model
x = torch.Tensor([-1.77130, -10.766500, 10.21840, -1.00430]).unsqueeze(0)

torch_out = torch_model(x).argmax(dim=1)[0]

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "note_detector.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


import onnx
import onnxruntime

onnx_model = onnx.load("note_detector.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("note_detector.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)[0].argmax()

print("Onnx output: ", type(ort_outs))
print("Torch Output: ", type(to_numpy(torch_out)))


# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-05, atol=1e-08)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

