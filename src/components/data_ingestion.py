import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import Transformation

from src.components.model_trainer import Model_trainer_config
from src.components.model_trainer import Model_trainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw.csv')

class DataIngest:
    def __init__(self):
        self.ingest_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('initiating the train_test split')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('started reading the data')

            os.makedirs(os.path.dirname(self.ingest_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingest_config.train_data_path, index=False, header=True)

            logging.info('Initated the train_test_split')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingest_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingest_config.test_data_path, index=False, header=True)

            logging.info('data ingestion completed')

            return (
                self.ingest_config.train_data_path,
                self.ingest_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngest()
    train_data,test_data = obj.initiate_data_ingestion()

    obj_1 = DataTransformationConfig()
    train_arr,test_arr = obj_1.datatransconfig(train_data,test_data)
    modeltrainer = Model_trainer()
    modeltrainer.model_development(train_arr,test_arr)