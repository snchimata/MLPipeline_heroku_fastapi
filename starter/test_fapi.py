import sys
import os


from fastapi.testclient import TestClient
<<<<<<< Updated upstream
from .main import app
=======
from . import main
>>>>>>> Stashed changes
import json

client = TestClient(main.app)

def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert json.loads(response.text)["Greeting"] == "Welcome to the Homepage of MLpipeline API!"

def test_post_50k():
    input_dict = {
                    "age": 23,
                    "workclass": "Private",
                    "fnlgt": 201490,
                    "education": "HS-grad",
                    "education_num": 9,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Adm-clerical",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Salary <= 50k"
    
def test_post_n50k():
    input_dict = {
                    "age": 48,
                    "workclass": "Self-emp-inc",
                    "fnlgt": 287927,
                    "education": "HS-grad",
                    "education_num": 9,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Wife",
                    "race": "White",
                    "sex": "Female",
                    "capital_gain": 25000,
                    "capital_loss": 0,
                    "hours_per_week": 20,
                    "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Salary > 50k"
