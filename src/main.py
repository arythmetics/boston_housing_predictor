from src.model.model import Model
from src.data.dataset import inputDataset

from typing import List
from fastapi import FastAPI
from fastapi import Depends
from pydantic import BaseModel, ValidationError, validator

import numpy as np
import pandas as pd

from fastapi import File, UploadFile, HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

class PredictRequest(BaseModel):
    data: List[List[float]]


class PredictResponse(BaseModel):
    data: List[float]


app = FastAPI()
model = Model()

#TODO: Find out how to feed the API inputs with a csv maybe
@app.post("/predict")
def predict(csv_file: UploadFile = File(...)):
    # return PredictResponse(data=[0.0]) Respond with PredictResponse structure

    try:
        df = pd.read_csv(csv_file.file).astype(float)
    except:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="Unable to process file"
        )
    
    df_n_instances, df_n_features = df.shape
    if df_n_features != model.n_features:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Each data point must contain {model.n_features} features",
        )

    y_pred = model.predict([0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98])
    y_pred = model.predict(df.to_numpy().reshape(-1, model.n_features))
    result = PredictResponse(data=[y_pred])

    return result
