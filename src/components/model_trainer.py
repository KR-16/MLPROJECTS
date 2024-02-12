import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the Training and Testing input data")
            X_Train, y_Train, X_test, y_Test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report: dict=evaluate_model(X_Train=X_Train, y_Train=y_Train, X_test=X_test, y_Test=y_Test, models=models)

            ### To get the best model score from the dictionary - report

            best_model_score = max(sorted(model_report.values()))

            ### To get the best model name from the dictionary - report

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model Found")
            
            logging.info("Best Model Found on Both Training and Testing Dataset")

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                object=best_model
            )

            predicted = best_model.predict(X_test)
            R2_Score = r2_score(y_Test, predicted)
            
            return R2_Score
        
        except Exception as e:
            raise CustomException(e,sys)