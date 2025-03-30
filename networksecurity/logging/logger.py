import os
import sys
import logging
import datetime as dt

log_file = f"{dt.datetime.now().strftime("%m_%d_%Y_%H_%M_S")}.log"
logs_path = os.path.join(os.getcwd(), "logs", log_file)
os.makedirs(logs_path, exist_ok = True)

log_file_path = os.path.join(logs_path, log_file)

logging_str = "[ %(asctime)s ]: %(lineno)d %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level = logging.INFO,
    format = logging_str,
    handlers= [
        logging.FileHandler(log_file_path), # To define the name where logs are saved
        logging.StreamHandler(sys.stdout) # To sure the logs are been shown in the terminal
    ]  
)

logger = logging.getLogger("security_network_project")