import os,sys
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from src.logger import logging
import pickle
import numpy as np
import dill

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)  # type: ignore
    

def evalute_metrics(true,predict):
    score=r2_score(true,predict)
    rmse=np.sqrt(mean_squared_error(true,predict))
    mae=mean_absolute_error(true,predict)

    return score,mae,rmse

    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        logging.info("error occured in load object")
        raise CustomException(e,sys) #type:ignore
    

def evalute_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs=GridSearchCV(model,param_grid=para,cv=3,n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            score,mae,rmse=evalute_metrics(y_test,y_test_pred)

            test_model_score=score
            report[list(models.keys())[i]]=test_model_score
        
        return report
    except Exception as e:
        logging.info("Error occured in evalute model method")
        raise CustomException(e,sys) #type:ignore