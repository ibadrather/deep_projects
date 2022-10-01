import onnxruntime as ort
from pydantic import BaseModel

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_prediction(data):
    # Loadin the ONNX model
    ort_session = ort.InferenceSession("bank_note_model.onnx")
    """
    Function to get predictions from ONNX model
    """
    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    return ort_outs.argmax()

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
