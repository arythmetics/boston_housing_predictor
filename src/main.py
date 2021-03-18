#debugging
from pathlib import Path
from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.dirname(Path(__file__).parent))

from src.model.model import Model, get_model
n_features = Model().n_inputs
from src.data.dataset import inputDataset

import uvicorn
from typing import List
from fastapi import FastAPI
from fastapi import Depends
from pydantic import BaseModel, ValidationError, validator

import numpy as np
import pandas as pd

from fastapi import File, UploadFile, HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from io import StringIO


class PredictRequest(BaseModel):
    data: List[float]

    @validator("data")
    def check_dimensionality(cls, i):
        if len(i) != n_features:
            raise ValueError(f"Each data point must contain {n_features} features")
        return i

class PredictResponse(BaseModel):
    data: List[float]


app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictRequest, model: Model = Depends(get_model)):
    X = np.array(input.data)
    y_pred = model.predict(X)
    y_pred = y_pred.astype('float32')
    result = PredictResponse(data=[y_pred])

    return result


@app.post("/predict_csv")
def predict_csv(csv_file: UploadFile = File(...), model: Model = Depends(get_model)):
    # AttributeError: 'SpooledTemporaryFile' object has no attribute 'readable'
    # Design change - this won't work for large files because TemporaryFile allows data to be stored to disc - this is all memory now
    bytes_data = csv_file.file.read()
    s = str(bytes_data,'utf-8')
    data = StringIO(s) 
    df = pd.read_csv(data)    

    y_pred = model.predict(df.to_numpy().reshape(-1, model.n_inputs))
    result = PredictResponse(data=y_pred.tolist())

    return result


if __name__ == '__main__':
    test_client = TestClient(app)
    data_path = Path(__file__).parent / "tests" / "api" / "data_correct.csv"
    with open(data_path, "r") as csv_file:
        response = test_client.post("/predict", files={"file": csv_file})
        print(response)
