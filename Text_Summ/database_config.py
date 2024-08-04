import pymongo
import pandas as pd
import numpy as np
import json
import os, sys
from dataclasses import dataclass
from Text_Summ.logger import logging


# @dataclass
# class EnvironmentVariable:
#     mongo_db_url:str = os.getenv("MONGO_DB_URL")

mongo_db_url ="mongodb+srv://aniket:aniket@cluster.rwhpc65.mongodb.net/"
#env_var = EnvironmentVariable()
mongo_client = pymongo.MongoClient(mongo_db_url)
logging.info(f"testing url {mongo_db_url} ")
logging.info(f"test {mongo_client}")
print(mongo_db_url)