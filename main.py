from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel

import tensorflow as tf
from src import models
from src.multiple_annotators_models import MA_GCCE
import numpy as np
from src.parameters import SENS_VARS_CHOC
app = FastAPI()

def GCCE_MA_loss(y_true, y_pred):
    return MA_GCCE.GCCE_MA_loss(y_true, y_pred)

def load_model_sens():
    models_sens = []
    for var in SENS_VARS_CHOC:
        models_sens.append(tf.keras.models.load_model(f"models/gcce/gcce_ma_{var}.keras",
                                                      custom_objects={'method': GCCE_MA_loss}))
    return models.MultiOutputModel(models_sens)

model_to_sens = load_model_sens()
model_to_fq = tf.keras.models.load_model('models/gcce/model_s2fq.keras')
# Load the models from the repository

@app.get("/predict_sens")
async def predict_sens(
    humedad: float = Query(...),
    grasa: float = Query(...),
    granulometria: float = Query(...),
    viscosidad: float = Query(...),
    fluidez: float = Query(...)
):
    input_data = np.array([[humedad, grasa,granulometria, viscosidad, fluidez]])
    predictions = model_to_sens.predict(input_data)
    return {"sens": predictions.tolist()}

@app.get("/predict_fq")
async def predict_fq(
    acidez: float = Query(...),
    amargor: float = Query(...),
    aroma: float = Query(...),
    astringencia: float = Query(...),
    dulce: float = Query(...),
    dureza: float = Query(...),
    impresion: float = Query(...),
    fusion: float = Query(...)
):
    input_data = np.array([[acidez, amargor, aroma, astringencia, dulce, dureza, impresion, fusion]])
    predictions = model_to_fq.predict(input_data)
    return {"fq": predictions.tolist()}