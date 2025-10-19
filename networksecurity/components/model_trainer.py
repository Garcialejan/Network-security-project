from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.evaluation import evaluate_models

import os
import sys
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from xgboost import XGBClassifier
from xgboost import XGBModel

import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse


import dagshub
from dotenv import load_dotenv
load_dotenv()
# dagshub.init(repo_owner='garcialejan', repo_name='Network-security-project', mlflow=True)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

class ModelTrainer:
    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classification_train_metric):
        mlflow.set_registry_uri(os.getenv(tracking_uri))
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        mlflow.set_experiment("Best model NetworkSecurity")
        
        with mlflow.start_run(run_name = "Best_model_params"):
            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Best model from hyperparameter-tunning with Grid Search")
                                  
            mlflow.log_metrics({
                "accuracy": classification_train_metric.accuracy_score,
                "f1_score": classification_train_metric.f1_score,
                "precision_score": classification_train_metric.precision_score,
                "recall_score": classification_train_metric.recall_score,
                })
            # I want to know if the model is a sklearn model or a xgb model
            model_name = best_model.__class__.__name__
            registered_model_name = model_name
            
            mlflow.sklearn.log_model(best_model, "model")
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                if isinstance(best_model, XGBModel):
                    # XGBoost model
                    mlflow.xgboost.log_model(
                        xgb_model=best_model,
                        artifact_path="model",
                        registered_model_name=registered_model_name
                    )
                elif isinstance(best_model, BaseEstimator):
                    # Scikit-learn model (DecisionTree, RandomForest, etc.)
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="model",
                        registered_model_name=registered_model_name
                    )
            else:
                if isinstance(best_model, XGBModel):
                    mlflow.xgboost.log_model(best_model, "model")
                elif isinstance(best_model, BaseEstimator):
                    mlflow.sklearn.log_model(best_model, "model")
                    
            # mlflow.log_param("best_n_estimators",best_params['n_estimators'])
            # mlflow.log_param("best_max_depth", best_params['max_depth'])
            # mlflow.log_param("best_min_samples_split", best_params['min_samples_split'])
            # mlflow.log_param("best_min_samples_leaf", best_params['min_samples_leaf'])
            # mlflow.log_metric("mse",mse)
          
            # signature=infer_signature(X_train, lr.predict(X_train))

        
    def train_model(self, X_train, y_train, X_test, y_test):
        '''
        Function to train the model with hyperparameter tunning
        and create the model trainer artifact for the predictions.
        '''
        models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(verbose=0),
                "Gradient Boosting": GradientBoostingClassifier(verbose=0),
                "Logistic Regression": LogisticRegression(verbose=0),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoostClassifier": XGBClassifier()
            }
        
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                'max_features':['sqrt','log2',None],
                'n_estimators': [32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                'n_estimators': [32,64,128,256],
                'max_depth': [3, 6, 10]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [32,64,128,256]
            },
            "XGBoostClassifier":{
                'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3],
                'max_depth':[3, 6, 10],
                'lambda':[0, 0.1, 1, 3],
                'alpha':[0, 0.1, 1, 3]
            }
            
        }
        
        # We use GridSearch to find the best model with the best hyperparameters
        model_report, best_params_per_model = evaluate_models(X_train=X_train,y_train=y_train,
                                                              X_test=X_test,y_test=y_test,
                                                              models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))
        ## To get best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        ## To get best model params from dict
        best_params = best_params_per_model[best_model_name]
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best score: {best_model_score:.3f}")
        
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        #* Track the trainning with MLFlow
        self.track_mlflow(best_model, classification_train_metric)
        
        y_test_pred=best_model.predict(X_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        #* Track the test with MLFlow
        self.track_mlflow(best_model, classification_test_metric)
        
        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok = True)

        Network_Model = NetworkModel(preprocessor = preprocessor, model = best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj = NetworkModel)
        
        # model pusher
        save_object("./final_models/model.pkl",best_model)
        
        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path = self.model_trainer_config.trained_model_file_path,
            train_metric_artifact = classification_train_metric,
            test_metric_artifact = classification_test_metric
            )
        logger.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
        
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
            
            model_trainer_artifact=self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)