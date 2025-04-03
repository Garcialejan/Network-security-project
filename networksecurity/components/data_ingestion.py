import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
## Configuration of the Data Ingestion COnfig
from networksecurity.entity.config_entity import DataIngestionConfig

from networksecurity.entity.artifact_entity import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()
username = os.getenv("db_username")
password = os.getenv("db_password")
if not username or not password:
    raise NetworkSecurityException("Database credentials are missing.", sys)
mongodb_uri = f"mongodb+srv://{username}:{password}@networkproject.y3uasyp.mongodb.net/?retryWrites=true&w=majority&appName=Networkproject"

class DataIngestion():
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    # 1 step: read data from MongoDB an transform into a df     
    def export_mongo_collection_as_df(self):
        '''
        Read data from MongoDB
        '''
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(mongodb_uri)
            collection = self.mongo_client[database_name][collection_name] # Query for the data into the collection
            df = pd.DataFrame(list(collection.find())) # Transform the query into a df
            if "id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis = 1)
            df = df.replace({"na":np.nan})
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    # 2 step: export data to feature store (in these case is a csv file)
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            # Creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    # 3 step: feature engineering and train test split
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logger.info("Performed train test split on the dataframe")

            logger.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
            logger.info(f"Exporting train and test file path.")
            
            # 4 step: ingest data into directory with .csv files
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logger.info(f"Exported train and test file path.")

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        
        
    def initiate_data_ingestion(self):
        try:
            df = self.export_mongo_collection_as_df()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.training_file_path,
                test_file_path =self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)