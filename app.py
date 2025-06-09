from fastapi import FastAPI, HTTPException
import pickle
from app.schemas import BankNote
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load the deepfake detection model
model = tf.keras.models.load_model("path_to_your_model.keras")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def root():
    return {"message": "Deepfake Detection API is up and running."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded video/image file
        contents = await file.read()
        # Dummy processing step (replace with your actual preprocessing)
        input_data = np.array([contents])  # Adjust as needed

        # Perform prediction
        predictions = model.predict(input_data)
        fake_prob = float(predictions[0][0])  # Example prediction logic
        is_fake = fake_prob > 0.5

        return PredictionResponse(
            prediction="Fake" if is_fake else "Real",
            confidence=fake_prob if is_fake else 1 - fake_prob
        )
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
