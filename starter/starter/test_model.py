# Script to test machine learning model.

import pandas as pd
import pytest
import joblib
import os
from .ml.data import *
from .ml.model import *

@pytest.fixture
def root():
    path = os.getcwd()
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    print(git_root)
    
    return git_root

@pytest.fixture
def data():
    """ Load initial data """
    data = pd.read_csv('./starter/data/cleaned_census.csv')
    return data

@pytest.fixture
def X_test():
    """ Load test data """
    return joblib.load("./starter/data/X_test.pkl")

@pytest.fixture
def y_test():
    """ Load test data """
    return joblib.load("./starter/data/y_test.pkl")

@pytest.fixture
def model():
    """ Load a pretrained model """
    return joblib.load("./starter/model/model.pkl")

#Test to process data function
def test_preprocess_data(data):
    """ Check to see if there are no . """
    try:    
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    except:
        assert False
    assert len(X)>0
    assert len(X)==len(y)

#Test model info
def test_model(model):
    try:
        model.best_estimator_
    except:
        assert False
    assert True

#Test inference length and compare to input data  
def test_inference(X_test):
    model = joblib.load("./starter/model/model.pkl")
    preds = inference(model, X_test)
    assert len(preds) > 0
    assert len(preds) == X_test.shape[0]
    
#Test metrics
def test_model_metrics(X_test,y_test):
    model = joblib.load("./starter/model/model.pkl")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert fbeta > 0.5, f"Performance is too low. fbeta = {fbeta} <= 0.5 "    
