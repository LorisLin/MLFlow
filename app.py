from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import numpy as np
import os

class PredictionRequest(BaseModel):
    features: list

app = FastAPI()

mlflow_tracking_uri = "http://host.docker.internal:8080"
model_name = "tracking-quickstart"
current_model = None
current_model_version = "1" 
next_model_version = current_model_version

def load_model(version):
    global current_model
    mlflow.set_tracking_uri(uri="http://host.docker.internal:8080")
    current_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")

load_model(current_model_version)
next_model = current_model

@app.post("/predict")
async def predict(request: PredictionRequest):
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features_array = np.array([request.features]) 
    predictions = current_model.predict(features_array).tolist()

    return {"predictions": predictions}

@app.post("/update-model")
async def update_model():
    global next_model, next_model_version
    try:
        next_model_version = str(int(next_model_version) + 1)
        next_model = load_model(next_model_version)
        return {"status": "Model updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/accept-next-model")
async def accept_next_model():
    global current_model, next_model, current_model_version, next_model_version
    try:
        current_model = next_model
        current_model_version = next_model_version
        return {"status": f"Accept next model with following verions: {current_model_version}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))