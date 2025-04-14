from networksecurity.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp # Used to detect the data drift

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.utils.main_utils.utils import read_yaml_file


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH) # Private atribute
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    @staticmethod()
    # Un método estático no tiene acceso ni a la instancia ni a la clase. 
    # Es simplemente una función que reside dentro de una clase. Se trata
    # de un método que pertenece a la clase pero no tiene acceso ni a la 
    # instancia (self) ni a la clase (cls).
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.DataFrame(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_columns(self, df: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config)
            logger.info(f"Requiered number of columns: {number_of_columns}")
            logger.info(f"Dataframe has columns: {len(df.columns)}")
            if len(df.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            # Read the date from train and test
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)
            
            # Validate number of columns
            status = self.validate_number_columns(df=train_df)
            if not status:
                error_message = "Train df does not contain all the columns.\n"
            status = self.validate_number_columns(df=test_df)
            if not status:
                error_message = "Test df does not contain all the columns.\n"
        except Exception as e:
            raise NetworkSecurityException(e, sys)