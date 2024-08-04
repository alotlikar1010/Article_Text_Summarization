import os ,sys

ARTIFACTS_DIR: str ="artifacts"
SOURCE_DIR_NAME: str= "Summarizer"

#Root Directory
ROOT_DIR = os.getcwd()

FILENAME='data.csv'
# Config File path 
CONFIG_DIR='config'
CONFIG_FILE_NAME='config.yaml'
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)


DATA_INGESTION_DATABASE_NAME= "textsumm"
DATA_INGESTION_COLLECTION_NAME= "collection"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"

DOWNLOAD_URL ="https://github.com/alotlikar1010/branching/blob/main/text_summarization.zip"
# common files
METADATA_DIR = "metadata"
METADATA_FILE_NAME: str = "metadata.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# constants related to data ingestion
DATA_INGESTION_ARTIFACTS_DIR: str = "data_ingestion_artifacts"
RAW_DATA_DIR_NAME: str = "raw_data"
DATA_INGESTION_TRAIN_DIR: str = "train"
DATA_INGESTION_TEST_DIR: str = "test"