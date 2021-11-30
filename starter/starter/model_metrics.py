# Script to evaluate machine learning model.

import joblib
from ml.model import *

#load data and model
X_train = joblib.load("../data/X_train.pkl")
X_test = joblib.load("../data/X_test.pkl")
y_train =joblib.load("../data/y_train.pkl")
y_test = joblib.load("../data/y_test.pkl")
model = joblib.load("../model/model.pkl")

#function to evaluate model
def model_metrics(model, X, y, description):
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    with open('../screenshots/model_metrics_output.txt', 'a') as f:
        f.write(f'{description}\n')
        f.write('\n')
        f.write('precision: ')
        f.write(str(precision))
        f.write('\n')
        f.write('recall: ')
        f.write(str(recall))
        f.write('\n')
        f.write('fbeta: ')
        f.write(str(fbeta))
        f.write('\n\n')
    
    f.close()

with open('../screenshots/model_metrics_output.txt', 'w') as f:
    pass
model_metrics(model, X_train, y_train, "Training Set Metrics")
model_metrics(model, X_test, y_test, "Test Set Metrics")
