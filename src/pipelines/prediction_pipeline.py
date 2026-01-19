import os,sys
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)

            predict=model.predict(data_scaled)
            return predict
        except Exception as e:
            logging.info("error occured in predict function in prediction pipeline.")
            raise CustomException(e,sys)#type:ignore

class CustomData:

    def __init__(self
                ,cut:str,
                color:str,
                clarity:str,
                carat:float,
                depth:float,
                table:float,
                x:float,
                y:float,
                z:float
                 ):
        self.cut=cut
        self.color=color
        self.clarity=clarity
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z

    def get_data_as_dataframe(self):
        try:
            custom_data_input={
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity],
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z]

            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            logging.info("error occured in data transform as dataframe")
            raise CustomException(e,sys)#type:ignore