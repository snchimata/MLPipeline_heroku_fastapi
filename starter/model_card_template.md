# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Author: Sai Chimata
- Date: December 01,2021
- Model version: 1.0
- Model: Random Forest Classifier
- Parameter grid: {'bootstrap': [True, False],
               'max_depth': [3, 5, 10, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1,2, 4, 8],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [25, 50, 100, 200]}
- For comments, email snctest@opayq.com

## Intended Use
    - This model was developed as one of the Udacity Machine Learning DevOps Nano Degree projects, which allowed learners to apply skills including developing a machine learning pipeline and deploying it in real time leveraging FastAPI, AWS S3, DVC, and Git

Primiary Intended Users: 
    - Academic training 

Out-of-scope Uses:
    - large-scale datasets
    - Realtime usecases



## Training Data
- Data was obtained from the UCI Machine Learning Repository  (https://archive.ics.uci.edu/ml/datasets/census+income)
- Data is cleaned and stored on S3
- 80% of data was randomly chosen for training

## Evaluation Data
- Remaining 20% of dataset was used for evaluation

## Metrics
The metrics used are:
  - fbeta
  - precision
  - recall
  
Training Set Metrics
precision: 0.9053276406162984
recall: 0.7744957916468159
fbeta: 0.8348168435467305

Test Set Metrics
precision: 0.7526881720430108
recall: 0.6347150259067358
fbeta: 0.6886858749121575

## Ethical Considerations
- Majority of data being from a single race impacts model performance of other groups.

## Caveats and Recommendations
- Model can be optimized by leveraging larger datasets with more features.
