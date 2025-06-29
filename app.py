"""
Created on 2025-03-17
Author: Charalampos Bekiaris

Example usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import pandas as pd
from pydantic import BaseModel
from models.xgb_reg import XGBModel as Model
from typing import List, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.encoders import jsonable_encoder

class Input(BaseModel):
    pickup_datetime: str
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
app = FastAPI()
model = Model().load_model('models/hp_xgbreg.xz')

@app.post("/predict")
async def predict(input_data: Union[Input, List[Dict[str, Any]]] = Body(...)):
    """
    Handles the prediction process for provided input data.
    Can accept either a single input object or a list of input objects.
    
    :param input_data: The input data required for predictive analysis.
    :return: A dictionary with the prediction results.
    """
    try:
        # Handle single input
        if not isinstance(input_data, list):
            features = pd.DataFrame([jsonable_encoder(input_data)])
            prediction = model.predict(features)
            print(prediction)
            return {"prediction": prediction.tolist()}
        
        # Handle list of inputs
        else:
            # Validate each item against the Input model
            validated_inputs = [Input(**item) for item in input_data]
            features = pd.DataFrame([jsonable_encoder(item) for item in validated_inputs])
            predictions = model.predict(features)
            return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")