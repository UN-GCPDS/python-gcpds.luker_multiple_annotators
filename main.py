from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import sqlite3

from src import models
from src.multiple_annotators_models import MA_GCCE
from src.parameters import SENS_VARS_CHOC, FQ_VARS_CHOC

app = FastAPI()
# %% load models
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

# %% db connection
def get_db_connection():
    conn = sqlite3.connect('data/chocolate/databases/chocolate_new.db')
    return conn

# %% schemas
class FqInput(BaseModel):
    cod_sampler: str
    humedad: float
    grasa: float
    granulometria: float
    viscosidad: float
    fluidez: float

class SensInput(BaseModel):
    cod_sampler : str
    acidez: float
    amargor: float
    aroma: float
    astringencia: float
    dulce: float
    dureza: float
    impresion: float
    fusion: float

# %% prediction endpoints
@app.post("/predict_sens")
async def predict_sens(data: FqInput):
    input_data = np.array([[data.humedad, data.grasa, data.granulometria,
                            data.viscosidad, data.fluidez]])
    
    predictions, _ = model_to_sens.predict(input_data)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"INSERT INTO train_fq (cod_sampler, {', '.join(FQ_VARS_CHOC)}) VALUES (?, ?, ?, ?, ?, ?)",
        (data.cod_sampler, data.humedad, data.grasa, data.granulometria, data.viscosidad, data.fluidez)
    )

    pred_values = predictions.flatten().tolist()
    cursor.execute(
        f"INSERT INTO pred_sens (cod_sampler, {', '.join(SENS_VARS_CHOC)}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (data.cod_sampler, *pred_values)
    )
    
    conn.commit()
    conn.close()

    return {"sens_predictions": dict(zip(SENS_VARS_CHOC, pred_values))}

@app.post("/predict_fq")
async def predict_fq(data: SensInput):
    input_data = np.array([[data.acidez, data.amargor, data.aroma, data.astringencia,
                            data.dulce, data.dureza, data.impresion, data.fusion]])
    predictions = model_to_fq.predict(input_data)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"INSERT INTO pred_sens (cod_sampler, {', '.join(SENS_VARS_CHOC)}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (data.cod_sampler, data.acidez, data.amargor, data.aroma, data.astringencia, data.dulce, data.dureza, data.impresion, data.fusion)
    )

    pred_values = predictions.flatten().tolist()
    cursor.execute(
        f"INSERT INTO train_fq (cod_sampler, {', '.join(FQ_VARS_CHOC)}) VALUES (?, ?, ?, ?, ?, ?)",
        (data.cod_sampler, *pred_values)
    )
    
    conn.commit()
    conn.close()

    return {"fq_predictions": dict(zip(FQ_VARS_CHOC, pred_values))}
