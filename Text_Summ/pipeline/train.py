from Text_Summ.components.data_ingestion import DataIngestion
from Text_Summ.entity.config_entity import *
from Text_Summ.entity.artifact_entity import *
from Text_Summ.logger import logging
from Text_Summ.exception import CustomException


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion in training pipeline")
        try: 
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion step completed successfully in train pipeline")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def run_pipeline(self) -> None:
        logging.info(">>>> Initializing training pipeline <<<<")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info("<<<< Training pipeline completed >>>>")
        except Exception as e:
            raise CustomException(e, sys)