import os
import torch
import numpy as np
import pandas as pd
from model import MLP
from ptl_modules import BankNoteClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import onnxruntime as ort

try:
    os.system("clear")
except:
    pass

# Loading the state dict into the model
net = MLP(4, 2)
torch_model = BankNoteClassifier(net)
state_dict = torch.load('bank_note_model.ckpt')["state_dict"]
torch_model.load_state_dict(state_dict)

# Model in Evaluation Mode
torch_model.eval()

# Load Test Data
test_data = pd.read_csv('BankNote_Authentication_test.csv')
print("Test Data shape: ", test_data.shape)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.astype(np.int32)

predicted_pt = []
actual_pt = []
for x, y in zip(X_test, y_test):
    x = torch.Tensor(x)
    y = torch.Tensor([y])
    output = torch_model(x)
    #print("Predicted: ", output.argmax().item(), "Actual: ", y.item())
    predicted_pt.append(output.argmax().item())
    actual_pt.append(y.item())

print("Claasification Report (PyTorch): \n", classification_report(actual_pt, predicted_pt))

print("Accuracy Score (PyTorch): ", accuracy_score(actual_pt, predicted_pt))
# print("Confusion Matrix: ", confusion_matrix(actual, predicted))

tn, fp, fn, tp = confusion_matrix(actual_pt, predicted_pt).ravel()
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)
print("True Positive: ", tp)

# Now we will convert to ONNX
# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "bank_note_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print("Model converted to ONNX")

# Let' test our converted model if accuracy is same

ort_session = ort.InferenceSession("bank_note_model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

predicted_onnx = []
actual_onnx = []
for x, y in zip(X_test, y_test):
    x = torch.Tensor(x)
    y = torch.Tensor([y])

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)[0]

    #print("Predicted: ", output.argmax().item(), "Actual: ", y.item())
    predicted_onnx.append(ort_outs.argmax())
    actual_onnx.append(y.item())


print("Accuracy Score (ONNX): ", accuracy_score(actual_onnx, predicted_onnx))

tn, fp, fn, tp = confusion_matrix(actual_onnx, predicted_onnx).ravel()
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)
print("True Positive: ", tp)

print("Claasification Report (PyTorch): \n", classification_report(actual_onnx, predicted_onnx))

# Result: Everything is same here. ONNX is working fine.