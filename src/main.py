from src.model.model import Model
from src.data.dataset import inputDataset

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel



class PredictRequest(BaseModel):
    data: List[List[float]]


class PredictResponse(BaseModel):
    data: List[float]


app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictRequest): #POST request with the PredictRequest structure
    return PredictResponse(data=[0.0]) #Respond with PredictResponse structure
