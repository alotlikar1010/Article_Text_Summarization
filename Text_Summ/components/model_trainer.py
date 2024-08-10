from Text_Summ.constant import *
from Text_Summ.entity.artifact_entity import *
from Text_Summ.entity.config_entity import *
from Text_Summ.exception import CustomException
from Text_Summ.logger import logging
from Text_Summ.exception import CustomException
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

class ModelTrainer:
    
    def __init__(self, model_trainer_config: ModelTrainerConfig, model, dataloader):
        self.model_trainer_config = model_trainer_config
        self.model = model
        self.dataloader = dataloader       
        
    
    def get_trainer_setup(self):
        try:
            logging.info("get trainer setup method started")
            checkpoint_dir = self.model_trainer_config.checkpoint_dir
            checkpoint_fname = self.model_trainer_config.checkpoint_fname
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=checkpoint_fname,
                save_top_k=1,
                monitor='val_loss',
                mode ='min'
            )
            logging.info("custom callbacks created")
            self.lighting_logger = TensorBoardLogger("pytorch_lighting_logs", name='summarizer_model')
            self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1, verbose=False, mode='min')
            logging.info('early_stopping_callback created')

            logging.info("get_trainer_setup method completed")           
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def get_trainer_object(self):
        try:
            logging.info("get trainer object method started")
            max_epochs = self.model_trainer_config.epochs
            
            if torch.cuda.is_available():
                trainer = pl.Trainer(check_val_every_n_epoch=1, max_epochs=max_epochs, accelerator='gpu',logger= self.lighting_logger,
                                    callbacks=[self.early_stopping_callback, self.checkpoint_callback]
                                    )
            else:
                trainer = pl.Trainer(check_val_every_n_epoch=1, max_epochs=max_epochs,logger= self.lighting_logger,
                                    callbacks=[self.early_stopping_callback, self.checkpoint_callback]
                                    )
            logging.info("get_trainer_object method completed")

            return trainer
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_model_trainer(self):
        
        try:
            logging.info("Model Training started")
            self.get_trainer_setup()
            trainer = self.get_trainer_object()
            
            torch.cuda.empty_cache()
            trainer.fit(self.model, self.dataloader)
            logging.info("model fine tunning done")
            
            trained_model_loss = trainer.callback_metrics
            trained_model_loss = float(trained_model_loss['val_loss'])
            trained_model_path = self.model_trainer_config.trained_model_dir
            os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
            torch.save({'model_state_dict': self.model.state_dict(),
                        'loss': trained_model_loss}, trained_model_path)
            
            model_trainer_artifacts = ModelTrainerArtifacts(model_loss=trained_model_loss, 
                                                            trained_model_path=trained_model_path
                                                            )
            logging.info(f"model trainder artifact created {model_trainer_artifacts}")
            logging.info("Model training completed")
            return model_trainer_artifacts                                   
        
        
        except Exception as e:
            raise CustomException(e, sys)