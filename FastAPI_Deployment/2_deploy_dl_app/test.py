import torch
import onnxruntime as ort
import numpy as np

data = torch.tensor([0.0, 0.0, 0.0, 0.0])


def to_numpy(data):
   return data.detach().cpu().numpy() if data.requires_grad else data.cpu().numpy()


ort_session = ort.InferenceSession("bank_note_model.onnx")


ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
ort_outs = ort_session.run(None, ort_inputs)[0]
output = ort_outs.argmax()

print(output)
