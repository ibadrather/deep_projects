import torch
from fastapi import FastAPI
from api_utils import get_prediction, BankNote

# FastAPI app
app = FastAPI()


@app.get("/")
def index():
    return {"Message": "Go to /docs endpoint to get predictions"}


@app.post("/predict")
def predict(data: BankNote):
    """
    Function to predict bank note authenticity
    """
    data = [data.variance, data.skewness, data.curtosis, data.entropy]
    data = torch.Tensor(data)
    prediction = get_prediction(data)

    if prediction == 1:
        return {"Prediction": "Warning! The bank note is Fake"}
    else:
        return {"Prediction": ":) The bank note is Real"}

# uvicorn deep_banknote_api:app --reload

