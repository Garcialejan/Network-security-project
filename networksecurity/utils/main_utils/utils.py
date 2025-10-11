import os,sys
import yaml
import json
import pickle
import joblib
from pathlib import Path
import numpy as np
# from typing import str, int, float, Path
from typing import Any
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

# Ensure library is designed to simplify testing and validation 
# of function arguments, return values, and other aspects. Provides
# decorators and helper functions to #*enforce type annotations,
#* constraints, or conditions

# The ConfigBox package is a python functionality that allows you to access 
# dictionary keys as if they were attributes It simplifies working
# with nested dictionaries. #* Simplifies working with JSON or YAML data

@ensure_annotations
def read_yaml_file(path_to_yaml: Path) -> ConfigBox:
    """
    Function to read yaml file and returns
    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file) # Read the yaml file
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content) # To use dict as arguments
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
@ensure_annotations
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

@ensure_annotations
def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

    
@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file with joblib library

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data with joblib library

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data
        

# @ensure_annotations
# def create_directories(path_to_directories: list, verbose=True):
#     """
#     Function to create list of directories

#     Args:
#         path_to_directories (list): list of path of directories
#         ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
#     """
#     for path in path_to_directories:
#         os.makedirs(path, exist_ok=True)
#         if verbose:
#             logger.info(f"created directory at: {path}")

# @ensure_annotations
# def save_json(path: Path, data: dict):
#     """save json data

#     Args:
#         path (Path): path to json file
#         data (dict): data to be saved in json file
#     """
#     with open(path, "w") as f:
#         json.dump(data, f, indent=4)

#     logger.info(f"json file saved at: {path}")

# @ensure_annotations
# def load_json(path: Path) -> ConfigBox:
#     """load json files data

#     Args:
#         path (Path): path to json file

#     Returns:
#         ConfigBox: data as class attributes instead of dict
#     """
#     with open(path) as f:
#         content = json.load(f)

#     logger.info(f"json file loaded succesfully from: {path}")
#     return ConfigBox(content)