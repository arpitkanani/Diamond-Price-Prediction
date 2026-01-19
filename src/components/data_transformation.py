from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import OrdinalEncoder

## Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from src.utils import save_object



## data Transformation configuration
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


## data IngestionConfig class 
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            ## Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols=['cut','color','clarity'] 
            numerical_cols=['carat','depth','table','x','y','z']
            #numerical_cols=['carat','depth','table','x','y','z','price']

            ## Define the custom ranking for the ordinal variables
            cut_categories=['Fair','Good','Very Good','Premium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline Initiated")

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())

                ]
            )

            ## Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent'))
                    ,('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
                    ,('scaler',StandardScaler())
                ]
            )        

            ## combine both numerical and categorical pipeline
            preprocessor=ColumnTransformer(
                [
                    ('num pipeline',num_pipeline,numerical_cols),
                    ('cat pipeline',cat_pipeline,categorical_cols)
                ]
            )
            logging.info("Pipeline Completed")
            return preprocessor
           
        except Exception as e:
            logging.info("Error in Data Transformation")
            
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            #reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info("Obtaining preprocessor object")

            preprocessing_obj=self.get_data_transformer_object()
            if preprocessing_obj is None:
                raise RuntimeError("get_data_transformer_object returned None — ensure it constructs and returns a transformer")
            target_column_name='price'
            drop_columns=['id',target_column_name]

            ## independent and dependent feature splitting
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            # apply the transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            ) 

            logging.info("Preprocessor pickle file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                )


        except Exception as e:
            logging.info("Error in Data Transformation")
               