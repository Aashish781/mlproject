import os
import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        train_report = {}
        test_report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            parameters = params[list(models.keys())[i]]

            gs = GridSearchCV(model, parameters, cv=3)
            gs.fit(X_train, y_train)

            # model.fit(X_train, y_train) # Train model

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test,  y_test_pred)

            train_report[list(models.keys())[i]] = train_model_score
            test_report[list(models.keys())[i]] = test_model_score

        return train_report, test_report
    except Exception as e:
        raise CustomException(e, sys)
    

if __name__ == "__main__":
    pass

