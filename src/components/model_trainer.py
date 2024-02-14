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
            X_Train, y_Train, X_Test, y_Test = (
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
                # "K-Neighbors Regressor": KNeighborsRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest": [{
                    "criterion":["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators":[8,16,32,64,128,256]
                }],

                "Decision Tree": [{
                    "criterion":["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2", None]
                }],

                "Gradient Boosting": [{
                    # "loss":["squared_error", "absolute_error", "huber", "quantile"],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "criterion":["squared_error", "friedman_mse"],
                    # "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators":[8,16,32,64,128,256]
                }],

                "Linear Regression": [{}],

                # "K-Neighbour Regressor": [{
                #     "n_neighbors": [5, 7, 9, 11, 13],
                #     "weights": ["uniform", "distance"],
                #     "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                # }],
                "CatBoosting Regressor": [{
                    "depth": [6, 8, 10],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "iterations": [30, 50, 100]
                }],

                "AdaBoost Regressor": [{
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "loss": ["linear", "square", "exponential"],
                    "n_estimators":[8,16,32,64,128,256]
                }]

            }

            # params={
            #     "Decision Tree": {
            #         'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            #     },
            #     "Random Forest":{
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "Gradient Boosting":{
            #         'learning_rate':[.1,.01,.05,.001],
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "Linear Regression": {},
            #     # "K-Neighbour Regressor":{
            #     #     'n_neighbors':[5,7,9,11]
            #     # },
            #     "CatBoosting Regressor":{
            #         'depth': [6,8,10],
            #         'iterations': [30, 50, 100]
            #     },
            #     "AdaBoost Regressor":{
            #         'learning_rate':[.1,.01,0.5,.001],
            #         'n_estimators': [8,16,32,64,128,256]
            #     }
            # }



            model_report: dict=evaluate_model(X_Train=X_Train, y_Train=y_Train, X_Test=X_Test, y_Test=y_Test, models=models, params = params)

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

            predicted = best_model.predict(X_Test)
            R2_Score = r2_score(y_Test, predicted)
            
            return R2_Score
        
        except Exception as e:
            raise CustomException(e,sys)