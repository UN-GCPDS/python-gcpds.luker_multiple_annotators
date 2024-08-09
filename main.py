from fastapi import FastAPI, HTTPException
import pickle
import os
from pydantic import BaseModel

app = FastAPI()

# Load the models from the repository
models = {}
model_path = "models"  # Adjust the path if necessary
for model_file in os.listdir(model_path):
    if model_file.endswith('.pkl'):
        model_name = model_file.split('.')[0]
        with open(os.path.join(model_path, model_file), 'rb') as f:
            models[model_name] = pickle.load(f)

# Define the request body
class PredictionRequest(BaseModel):
    model_name: str
    data: list

# Define the predict endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    model = models.get(request.model_name)
    if model:
        prediction = model.predict([request.data])
        return {"prediction": prediction.tolist()}
    else:
        raise HTTPException(status_code=404, detail="Model not found")
