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

root = os.getcwd()
    
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


try:
    model = joblib.load(os.path.join(root,"./starter/model/model.pkl"))
except FileNotFoundError:
    model = joblib.load("./starter/model/model.pkl")

try:
    enc = joblib.load(os.path.join(root,"./starter/model/encoder.enc"))
except FileNotFoundError:
    enc = joblib.load("./starter/model/encoder.enc")

try:
    lb_enc = joblib.load(os.path.join(root,"./starter/model/lb.enc"))
except FileNotFoundError:
    lb_enc = joblib.load("./starter/model/lb.enc")


@app.get("/")
async def root():
    return {"Greeting": "Welcome to the Homepage of MLpipeline API!"}

@app.post("/predict", response_model=Output, status_code=200)
async def predict(data: Input):

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
                              
    prediction = inference(model, X)
    
    if prediction == 1:
        output = "Salary > 50k"
    else:
        output = "Salary <= 50k"
    return {"prediction": output}
