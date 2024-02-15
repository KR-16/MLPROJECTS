import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, object):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)
        
        with open(file=file_path, mode="wb") as file_object:
            dill.dump(object, file_object)
       
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_Train, y_Train, X_Test, y_Test, models, params):
    try:
        report = {}
        for m in range(len(list(models))):
            model = list(models.values())[m]
            para = params[list(models.keys())[m]]

            Gridsearch = GridSearchCV(model, para, cv = 3)
            Gridsearch.fit(X_Train, y_Train)

            model.set_params(**Gridsearch.best_params_)
            # Train Model
            model.fit(X_Train, y_Train)

            y_train_predict = model.predict(X_Train)

            y_test_predict = model.predict(X_Test)

            train_model_score = r2_score(y_Train, y_train_predict)

            test_model_score = r2_score(y_Test, y_test_predict)

            report[list(models.keys())[m]] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        with open(filepath, "rb") as file_object:
            return dill.load(file_object)
    except Exception as e:
        raise CustomException(e, sys)