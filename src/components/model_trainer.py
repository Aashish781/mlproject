import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_apth = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer (self, train_array, test_array):
        try:
            logging.info("Split training ans test input data")
            X_train, y_train = (train_array[:, :-1], train_array[:, -1])
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                # "XGB Classifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }
            
            train_model_report, test_model_report = evaluate_models(X_train=X_train, y_train=y_train,
                                               X_test = X_test, y_test=y_test, 
                                               models = models)
            
            logging.info(f"Model evalution completed and test result: {test_model_report}")

            ## To get best model score from dict
            best_model_score = max(sorted(test_model_report.values()))

            ## To get best model name from dict
            best_model_name = list(test_model_report.keys())[
                list(test_model_report.values()).index(best_model_score)]
            
            logging.info(f"Best model name: {best_model_name} with a score: {best_model_score}")
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_apth,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

