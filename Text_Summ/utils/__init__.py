import pandas as pd
from Text_Summ.logger import logging
from Text_Summ.exception import CustomException
import os , sys
from Text_Summ.database_config import mongo_client
#import yaml,dill
#import numpy as np

def get_collection_as_dataframe(database_name:str, collection_name: str)->pd.DataFrame:
    
    try:
        
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        logging.info("Error occur in get_collection_as_dataframe function")
        raise CustomException(e, sys)
    
