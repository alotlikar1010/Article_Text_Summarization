import os, sys
from dataclasses import dataclass
from datetime import datetime
from Text_Summ.constant import *

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# artifact | data_ingestion_timestap

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = SOURCE_DIR_NAME
    artifact_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    download_url: str = DOWNLOAD_URL
    file_name: str = METADATA_FILE_NAME
    # Mater Data Ingestion directory 
    data_ingestion_artifacts: str = os.path.join(ROOT_DIR, training_pipeline_config.artifact_dir, DATA_INGESTION_ARTIFACTS_DIR)
    
    # Data Ingestion Directory 
    raw_data_dir: str = os.path.join(ROOT_DIR, data_ingestion_artifacts, RAW_DATA_DIR_NAME)
    ingested_data: str = os.path.join(raw_data_dir,'ingested_data')
    #ingested_data_dir: str =os.path.join(raw_data_dir,data_ingestion_key[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
    
    
    # Split train and TEst Data
    train_file_path: str = os.path.join(data_ingestion_artifacts, DATA_INGESTION_TRAIN_DIR, TRAIN_FILE_NAME)
    test_file_path: str = os.path.join(data_ingestion_artifacts, DATA_INGESTION_TEST_DIR, TEST_FILE_NAME)