import os
import sys
import json
import pandas as pd
import numpy as np
import pymongo
from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException


from dotenv import load_dotenv
load_dotenv()
username = os.getenv("db_username")
password = os.getenv("db_password")
if not username or not password:
    raise NetworkSecurityException("Database credentials are missing.", sys)
mongodb_uri = f"mongodb+srv://{username}:{password}@networkproject.y3uasyp.mongodb.net/?retryWrites=true&w=majority&appName=Networkproject"

import certifi
# Se utiliza para verificar la identidad de sitios 
# web seguros (HTTPS) cuando realizas solicitudes 
# a través de internet. Certifi proporciona un 
# conjunto actualizado de certificados raíz de 
# confianza que Python puede usar para verificar 
# que los sitios web HTTPS sean seguros.
ca = certifi.where() 
# Devuelve la ubicación (path) en tu sistema donde 
# se almacena el archivo que contiene los certificados
# raíz de confianza. ca = certificate authorities.


class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def csv_to_json(self, data_path):
        try:
            df = pd.read_csv(data_path, sep = ",", index_col = False)
            records = list(json.loads(df.T.to_json()).values())
            return records
        except FileNotFoundError:
            raise NetworkSecurityException("File not found.", sys)
        except Exception as e:
            raise NetworkSecurityException(f"Error processing CSV: {e}", sys)

    
    def insert_data_mongodb(self, records, database, collection):
        """
        Take the data into the database

        Args:
            records (str): data
            database (str): databse name
            collection (str): is like the table name in SQL tables
        """
        try:
            self.records = records
            self.database = database
            self.collection = collection
            
            self.mongo_client = pymongo.MongoClient(mongodb_uri, tlsCAFile=ca)  # Usa certificados seguros
            # Connect python with mongo db. Python language
            # converts into actions to mongodb
            self.database = self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        
        except pymongo.errors.ConnectionFailure:
            raise NetworkSecurityException("Failed to connect to MongoDB.", sys)
        except Exception as e:
            raise NetworkSecurityException(f"Error inserting data into MongoDB: {e}", sys)
        
        
if __name__ =="__main__":
    FILE_PATH = "Network_Data/phisingData.csv"
    DATA_BASE = "NetworkProject"
    collection = "NetworkData"
    
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json(FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_mongodb(records,
                                                   database=DATA_BASE,
                                                   collection=collection)
    print(no_of_records)
