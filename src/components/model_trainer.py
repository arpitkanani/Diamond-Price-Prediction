from mlflow.models.model import urlparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.logger import logging
from src.exception import CustomException

from src.utils import save_object,evalute_metrics
from src.utils import evalute_model
from dataclasses import dataclass
import sys,os
import dagshub

dagshub.init(repo_owner='arpitkanani', repo_name='Diamond-Price-Prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "DecisionTree Regeressor":DecisionTreeRegressor(),
                "RandomForest Regressor":RandomForestRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "GradiantBoost Regressor":GradientBoostingRegressor(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor()
            }

            param = {
           "Linear Regression": {
            #"fit_intercept": [True, False]
            },

            "Lasso": {
                #"alpha": [0.001, 0.01, 0.1, 1, 10],
                #"max_iter": [1000, 5000]
            },

            "Ridge": {
                #"alpha": [0.1, 1, 10, 50],
                #"solver": ["auto", "svd", "cholesky"]
            },
            "DecisionTree Regeressor": {
                #"criterion": ["squared_error", "friedman_mse"],
                #"max_depth": [None, 5,3, 10],
                # "min_samples_split": [2, 5, 10],
                #"min_samples_leaf": [1, 2, 4]
            },

            "RandomForest Regressor": {
                #'n_estimators': [100, 200, 300],
                #'max_depth': [None, 10, 20, 30],
                #'max_features': ['sqrt', 'log2'],
                #'bootstrap': [True, False]
            },

            "AdaBoost Regressor": {
                #"n_estimators": [50, 100, 200],
                #"learning_rate": [0.01, 0.1, 1]
            },

            "GradiantBoost Regressor": {
                #"n_estimators": [100, 200],
                #"learning_rate": [0.01, 0.1],
                #"max_depth": [3, 5],
                #"subsample": [0.8, 1.0]
            },

            'XGB Regressor' : {
                #"n_estimators": [200, 400],
                #"learning_rate": [0.03, 0.05, 0.1],
                #"max_depth": [4, 6, 8],
                #"subsample": [0.7, 0.8, 1.0],
                #"colsample_bytree": [0.7, 0.8, 1.0],
                #"reg_lambda": [1, 5, 10]    
            },
            'CatBoost Regressor':{
                #"iterations": [300, 500],
                #"learning_rate": [0.03, 0.05, 0.1],
                #"depth": [4, 6, 8],
                #"l2_leaf_reg": [1, 3, 5, 7]
            }    
        }


            model_report:dict=evalute_model(X_train,y_train,X_test,y_test,models,param)

            print(model_report)
            print("\n====================================================================================\n")
            logging.info(f"Model Report : {model_report}")

            #to get best model score dictionary

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            print(f"Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}")
            print("\n====================================================================================\n")

            logging.info(f"Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}")

            models_names=list(param.keys())
            
            actual_model=""
            for model in models_names:
                if best_model_name == model:
                    actual_model=actual_model + model
            
            
            best_params=param[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/arpitkanani/Diamond-Price-Prediction.mlflow")
            tracking_url_type_store= urlparse(mlflow.get_tracking_uri()).scheme

            #ml flow pipe line
            with mlflow.start_run():
                predicted_qualities=best_model.predict(X_test)

                (score,rmse,mae)=evalute_metrics(y_test,predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("r2",score)
                mlflow.log_metric("mae",mae)
                

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(sk_model=best_model,name="model") # type: ignore

                else:
                    mlflow.sklearn.log_model(best_model,'model') # type: ignore

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            
        
        except Exception as e:
            logging.info("Error occurred in Model Training")
            raise CustomException(e,sys)  # type: ignore