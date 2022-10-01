import uvicorn  #ASGI
from fastapi import FastAPI 
from banknote import BankNote
import pickle

# Load model
with open("bank_note_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)


# Make FastAPI file
app = FastAPI()

@app.get("/")
def index():
    return {"Message": "Welcome to bank note classification FastAPI app."}


@app.get("/{name}")
def get_name(name: str):
     return {"Message": f"Welcome {name} to bank note classification FastAPI app."}


@app.post("/predict")
def predict_bank_note(data: BankNote):
    data = data.dict()
    data_predict = [[data["variance"], data["skewness"], data["curtosis"], data["entropy"]]]

    model_prediction = classifier.predict(data_predict)

    if model_prediction[0] == 1:
        return {"Model Prediction": "Warning! The note is fake!"}

    return {"Model Prediction": "Don't worry, the note is not fake."}

def run_app():
    config = uvicorn.Config("banknote_api:app", port=5000, log_level="info", reload=False)
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    run_app()
    