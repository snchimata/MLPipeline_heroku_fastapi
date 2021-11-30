# Script to evaluate machine learning model on sliced data.

import pandas as pd
import joblib
from ml.data import process_data
from ml.model import *

df = pd.read_csv("../data/cleaned_census.csv")
model = joblib.load("../model/model.pkl") 
encoder = joblib.load("../model/encoder.enc")
lb = joblib.load("../model/lb.enc")

def slice_metrics(df, cat_feature):
    """ Function for calculating performance on slices of the dataset."""
    for cls in df[cat_feature].unique():
        df_temp = df[df[cat_feature] == cls]
        X, y, _, _ = process_data(df_temp, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb)
        
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)
        
    with open('../screenshots/slice_metrics_output.txt', 'a') as f:
        f.write(f'metrics on slice data\ncategorical feature: {cat_feature}\n')
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

with open('../screenshots/slice_metrics_output.txt', 'w') as f:
    pass

for ftr in cat_features:
    print(ftr)
    slice_metrics(df, ftr)
