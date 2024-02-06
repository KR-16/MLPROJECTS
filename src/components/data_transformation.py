import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
# Imputor is used for handling missing values
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataInformationConfig:
    preprocessor_object_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataInformationConfig()

    def get_data_transformer_object(self):

        """

        This function is responsible for Data Transformation
        
        """
        try:
            # data = pd.read_csv("notebook\data\stud.csv")
            # numerical_features = [column for column in data.columns if data[column].dtype != "O"]
            # categorical_features = [column for column in data.columns if data[column].dtype == "O"]

            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical Columns Standard Scaling Completed", numerical_features)

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical Columns Encoding completed", categorical_features)

            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    ("categorical_pipeline", categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # data = pd.read_csv(raw_path)
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Reading Train and Test Data Completed")

            logging.info("Obtaining Preprocessing Object")

            preprocessing_object=self.get_data_transformer_object()

            target_column_name = "math_score"
            # numerical_features = [column for column in data.columns if data[column].dtypes != "O" and column == "math_score"]

            input_features_training_data = train_data.drop(columns=[target_column_name], axis = 1)
            target_feature_train_data = train_data[target_column_name]

            input_features_testing_data = test_data.drop(columns=[target_column_name], axis = 1)
            target_feature_test_data = test_data[target_column_name]

            logging.info("Applying Preprocessing Object on Training and Testing Data")

            input_features_training_array = preprocessing_object.fit_transform(input_features_training_data)
            input_features_testing_array = preprocessing_object.transform(input_features_testing_data)

            train_array = np.c_[
                input_features_training_array, np.array(target_feature_train_data)
            ]

            test_array = np.c_[
                input_features_testing_array, np.array(target_feature_test_data)
            ]

            logging.info("Saving Preprocessing Object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_object_file_path,
                object = preprocessing_object
            )

            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_object_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)