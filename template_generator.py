import os
from pathlib import Path
import logging

project_name = "networksecurity"
logging.basicConfig(level = logging.INFO,
                    format  ='[%(asctime)s]: %(message)s:')

list_of_files = [
    f"{project_name}/__init__.py", 
    f"{project_name}/components/__init__.py", 
    f"{project_name}/utils/__init__.py", 
    f"{project_name}/utils/common.py", 
    f"{project_name}/logging/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/exception/__init__.py", 
    f"{project_name}/entity/__init__.py", 
    f"{project_name}/entity/config_entity.py", 
    f"{project_name}/constants/__init__.py",
    f"{project_name}/cloud/__init__.py"
    "Network_Data/",
    "notebooks/",
    "main.py",
    "Dockerfile",
    # "requierements.txt",
    "setup.py", #! Use to create the entire project as a package
    ".env"
]

# Loop to create all fthe dirs and files above
for filepath in list_of_files:
    filepath = Path(filepath)
    file_dir, file_name = os.path.split(filepath)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory {file_dir} for the file: {file_name}")
    
    elif (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file {filepath}")
        
    else:
        logging.info(f"{file_name} already exist")