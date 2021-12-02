"""
Author: SChimata
Date: Dec 2, 2021
This script is used to implement ML pipeline using FastAPI
"""
import os
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import joblib

from .starter.ml.model import *
from .starter.ml.data import *
    
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    dvc_output = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    print(dvc_output.stdout)
    print(dvc_output.stderr)
    if dvc_output.returncode != 0:
        print("dvc pull failed")
    else:
        os.system("rm -r .dvc .apt/usr/lib/dvc")

    
class Input(BaseModel):
    age : int = 25
    workclass : str = 'Private'
    fnlgt : int = 201490
    education : str = 'Bachelors'
    education_num : int = 13
    marital_status : str ='Never-married' 
    occupation : str = 'Adm-clerical'
    relationship : str = 'Husband'
    race : str = 'White'
    sex : str = 'Male'
    capital_gain : int = 0
    capital_loss : int = 0
    hours_per_week : int = 40
    native_country : str = 'United States'

class Output(BaseModel):
    prediction: str
        
app = FastAPI()

model = joblib.load("./starter/model/model.pkl")
enc = joblib.load("./starter/model/encoder.enc")
lb_enc = joblib.load("./starter/model/lb.enc")

@app.get("/")
async def root():
    return {"Greeting": "Welcome to the Homepage of MLpipeline API!"}

@app.post("/predict", response_model=Output, status_code=200)
async def predict(data: Input):
    '''
    input_data = pd.DataFrame([{"age" : data.age,
                        "workclass" : data.workclass,
                        "fnlgt" : data.fnlgt,
                        "education" : data.education,
                        "education-num" : data.education_num,
                        "marital-status" : data.marital_status,
                        "occupation" : data.occupation,
                        "relationship" : data.relationship,
                        "race" : data.race,
                        "sex" : data.sex,
                        "capital-gain" : data.capital_gain,
                        "capital-loss" : data.capital_loss,
                        "hours-per-week" : data.hours_per_week,
                        "native-country" : data.native_country}])
    '''
    request_dict = data.dict(by_alias=True)
    input_data = pd.DataFrame(request_dict, index=[0])

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    
    X, _, _, _ = process_data(input_data, 
                                     categorical_features=cat_features, 
                                     training=False, 
                                     encoder = enc, 
                                     lb = lb_enc) 
    prediction_outcome = inference(model, X)
    
    if prediction_outcome == 1:
        pred = "Salary > 50k"
    else:
        pred = "Salary <= 50k"
    return {"prediction": pred}
