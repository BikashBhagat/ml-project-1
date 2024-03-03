'''
This file will have the output of data transform as input and delivered best suitable model as output
'''

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_object_path = os.path.join("artifact","model_object.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            # Divode tran and test
            X_train, y_train, X_test, y_test =(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )   
            logging.info("Train Test Split Completed")
            # Define the model which we will test 
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            #Evaluation Matrices
            logging.info("Model Training Started")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models)
            logging.info("Model Training Completed")

            # Get the best model score and name
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            
            if best_model_score>0.6:
                logging.info(f"Best Model found: {best_model_name} R2_Score: {best_model_score}")
                print(f"Best Model found: {best_model_name} R2_Score: {best_model_score}")
            else:
                logging.info("No Best model found")
                print("No Best model found")
            
            # Save the model in pickel file 
            save_object(file_path=self.model_trainer_config.model_object_path, obj=best_model)

            predicted_y= best_model.predict(X_test)

            score=r2_score(y_test,predicted_y)

            print(score)
            return score
        
        except Exception as e:
            raise CustomException(e,sys)