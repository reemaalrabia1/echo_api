from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# تحميل المودل
model = joblib.load("echo_model.pkl")

# تعريف شكل البيانات اللي بتجي من الفرونت
class AudioFeatures(BaseModel):
    time: float
    sound_level: float

@app.get("/")
def home():
    return {"message": "EchoFans API is running!"}

@app.post("/predict")
def predict(data: AudioFeatures):
    features = np.array([[data.time, data.sound_level]])
    prediction = model.predict(features)
    result = "حماس" if prediction[0] == 1 else "هادئ"
    return {
        "prediction": int(prediction[0]),
        "result": result
    }