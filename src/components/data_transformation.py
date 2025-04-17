import os
import sys
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object



@dataclass

class DataTransformationConfig:
    def __init__(self):
        self.datatransconfig =os.path.join('artifacts','preprocessor.pkl')

class Transformation:

    def __init__(self):
        self.transconfig = DataTransformationConfig()

    def transformings(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[('imputer',SimpleImputer(strategy='median')),('stdscaler',StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy=('most frequent'))),
                    ('stdscaler',StandardScaler()),
                    ('encoder',OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                ('num_pipelines',num_pipeline,numerical_columns),
                ('cat_pipelines',cat_pipeline,categorical_columns)
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiated_transformations(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            preprocessing_obj = self.transformings
            target_column = 'math_score'
            input_train_df = train_df.drop(columns=[target_column],axis=1)
            target_train_df = train_df[target_column]

            input_test_df = test_df.drop(columns=[target_column],axis=1)
            target_test_df = test_df[target_column]

            train_df_arr = preprocessing_obj.fit_transform(input_train_df)
            test_df_arr = preprocessing_obj.fit(input_test_df)

            train_arr = np.c_[train_df_arr,np.array(target_train_df)]

            test_arr = np.c_[test_df_arr,np.array(target_test_df)]

            return(
                train_arr,
                test_arr,
                self.transconfig.datatransconfig
            )
        
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
        
        except Exception as e:
            raise CustomException(e,sys)

