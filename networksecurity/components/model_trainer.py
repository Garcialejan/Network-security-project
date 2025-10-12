from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
# from networksecurity.utils.ml_utils.model.evaluation import evaluate_models

import os
import sys
from pathlib import Path


class ModelTrainer:
    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def train_model(self, X_train, y_train):
        pass
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try: 
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            # loading train and test arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            model = self.train_model(X_train, y_train)
        except Exception as e:
            raise NetworkSecurityException(e, sys)