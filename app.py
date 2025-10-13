import os
import sys
from fastapi import FastAPI
import pymongo
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
# Se utiliza para manejar y permitir solicitudes HTTP desde orígenes  distintos al del servidor.
# CORS es un mecanismo de seguridad implementado en los navegadores web que bloquea solicitudes 
# HTTP entre distintos orígenes. Sin embargo, en muchos casos (como cuando tu frontend está en 
# un dominio y tu API en otro), necesitas permitir explícitamente esas solicitudes cruzadas

from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
# RedirectResponse se utiliza para devolver una respuesta HTTP que redirige al cliente (navegador)
# a otra URL. Esta clase genera una respuesta con: Código de estado HTTP típicamente 302 (Found) y
# una cabecera Location que contiene la URL de destino.

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.pipeline.trainning_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

import certifi
# Se utiliza para verificar la identidad de sitios  web seguros (HTTPS) cuando realizas solicitudes 
# a través de internet. Certifi proporciona un conjunto actualizado de certificados raíz de confianza
# que Python puede usar para verificar que los sitios web HTTPS sean seguros.
ca = certifi.where() 
# Devuelve la ubicación (path) en tu sistema donde se almacena el archivo que contiene los certificados
# raíz de confianza. ca = certificate authorities.

# Create the connection to mongodb
from dotenv import load_dotenv
load_dotenv()
username = os.getenv("db_username")
password = os.getenv("db_password")
if not username or not password:
    raise NetworkSecurityException("Database credentials are missing.", sys)
mongodb_uri = f"mongodb+srv://{username}:{password}@networkproject.y3uasyp.mongodb.net/?retryWrites=true&w=majority&appName=Networkproject"

client = pymongo.MongoClient(mongodb_uri, tlsCAFile=ca)

from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Start the FastAPI app creation
app = FastAPI()
origins = ["*"]

app.add_middleware(CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Se permiten solicitudes desde cualquier dominio, se permite que el navegador 
# envie credenciales (cookies, autenticación), se permiten todos los métodos 
# HTTP (GET, POST, PUT, DELETE) y se permite cualquier encabezado en las solicitudes
# del cliente. Permitir todos los origenes y el envio de credenciales a la vez
# puede representar un riesgo de seguridad, ya cualquiera puede acceder a datos
# privados de usuario por eso es recomendable no permitir cualquier origen.

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
if __name__ == "__main__":
    app_run(app, host="127.0.0.1", port=8000)