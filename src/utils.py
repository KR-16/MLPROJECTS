import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, object):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)
        
        with open(file=file_path, mode="wb") as file_object:
            dill.dump(object, file_object)
        
    except Exception as e:
        raise CustomException(e,sys)