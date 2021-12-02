"""
Author: SChimata
Date: Dec 3, 2021
This script is used to test live api
"""
import requests
import json 

data1 = {
                "age": 21,
                "workclass": "Private",
                "fnlgt": 201490,
                "education": "9th",
                "education_num": 5,
                "marital_status": "Married-civ-spouse",
                "occupation": "Adm-clerical",
                "relationship": "Wife",
                "race": "White",
                "sex": "Female",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 20,
                "native_country": "United-States"
                }

data2 = {
                "age": 49,
                "workclass": "Self-emp-inc",
                "fnlgt": 287927,
                "education": "Masters",
                "education_num": 14,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 30000,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
                }

response = requests.post('https://mlpipeline-app.herokuapp.com/predict/', data=json.dumps(data1))
print(response.status_code)
print(response.json())

response = requests.post('https://mlpipeline-app.herokuapp.com/predict/', data=json.dumps(data2))
print(response.status_code)
print(response.json())
