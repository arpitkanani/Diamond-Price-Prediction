import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


#initialize the Data Ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

#create a data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method Starts")

        try:
            df=pd.read_csv(os.path.join('notebooks/data','Diamond.csv'))
            logging.info("Data read as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train Test Split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of Data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error occurred in Data Ingestion configuration")
