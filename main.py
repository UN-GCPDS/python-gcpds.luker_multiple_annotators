from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import sqlite3
import pandas as pd

from src import models
from src.multiple_annotators_models import MA_GCCE
from src.utils import get_iAnn
from src.parameters import *

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
# OLD_DATA = "data/chocolate/databases/chocolate.db"
# NEW_DATA = "data/chocolate/databases/chocolate_new.db"

# def get_db_connection(db_path):
#     conn = sqlite3.connect(db_path)
#     return conn

# def read_from_db(db_path, table_name):
#     conn = sqlite3.connect(db_path)
#     query = f"SELECT * FROM {table_name}"
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     return df

# %% schemas
# class FqInput(BaseModel):
#     cod_sampler: str
#     humedad: float
#     grasa: float
#     granulometria: float
#     viscosidad: float
#     fluidez: float

class FqInput(BaseModel):
    humidity_halogen : float
    fat_nmr : float
    granulometry_micrometer : float
    plastic_viscosity_anton_paar : float
    flow_limit_anton_paar : float
    reference_id : float

# class SensInput(BaseModel):
#     cod_sampler : str
#     acidez: float
#     amargor: float
#     aroma: float
#     astringencia: float
#     dulce: float
#     dureza: float
#     impresion: float
#     fusion: float

class SensInput(BaseModel):
    acidity : float
    bitterness : float
    aroma : float
    astringency : float
    sweetness : float
    hardness : float
    global_impression : float
    melting_speed : float
    reference_id : float

# %% prediction endpoints
@app.post("/predict_sens")
async def predict_sens(data: FqInput):
    input_data = np.array([[data.humidity_halogen, data.fat_nmr, data.granulometry_micrometer,
                            data.plastic_viscosity_anton_paar, data.flow_limit_anton_paar]])
    
    predictions, _ = model_to_sens.predict(input_data)
    # conn = get_db_connection(NEW_DATA)
    # cursor = conn.cursor()

    # cursor.execute(
    #     f"INSERT INTO train_fq (cod_sampler, {', '.join(FQ_VARS_CHOC)}) VALUES (?, ?, ?, ?, ?, ?)",
    #     (data.cod_sampler, data.humedad, data.grasa, data.granulometria, data.viscosidad, data.fluidez)
    # )

    pred_values = predictions.flatten().tolist()
    # cursor.execute(
    #     f"INSERT INTO pred_sens (cod_sampler, {', '.join(SENS_VARS_CHOC)}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    #     (data.cod_sampler, *pred_values)
    # )
    
    # conn.commit()
    # conn.close()
   
    return  {TRANS_SENS_VARS_CHOC[SENS_VARS_CHOC[i]] : pred_values[i] for i in range(len(pred_values))}

@app.post("/predict_fq")
async def predict_fq(data: SensInput):
    input_data = np.array([[data.acidity, data.bitterness, data.aroma, data.astringency,
                            data.sweetness, data.hardness, data.global_impression, data.melting_speed]])
    predictions = model_to_fq.predict(input_data)

    # conn = get_db_connection(NEW_DATA)
    # cursor = conn.cursor()

    # cursor.execute(
    #     f"INSERT INTO pred_sens (cod_sampler, {', '.join(SENS_VARS_CHOC)}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    #     (data.cod_sampler, data.acidez, data.amargor, data.aroma, data.astringencia, data.dulce, data.dureza, data.impresion, data.fusion)
    # )

    pred_values = predictions.flatten().tolist()
    # cursor.execute(
    #     f"INSERT INTO train_fq (cod_sampler, {', '.join(FQ_VARS_CHOC)}) VALUES (?, ?, ?, ?, ?, ?)",
    #     (data.cod_sampler, *pred_values)
    # )
    
    # conn.commit()
    # conn.close()
    return  {TRANS_FQ_VARS_CHOC[FQ_VARS_CHOC[i]] : pred_values[i] for i in range(len(pred_values))}


# @app.post("/retrain_model_fq")
# async def retrain_model_fq():
#     old_df_sens = read_from_db(OLD_DATA, 'pred_sens')
#     new_df_sens = read_from_db(NEW_DATA, 'pred_sens')
#     df_sens = pd.concat([old_df_sens, new_df_sens], ignore_index=True).set_index('cod_sampler')

#     old_df_fq = read_from_db(OLD_DATA, 'train_fq')
#     new_df_fq = read_from_db(NEW_DATA, 'train_fq')
#     df_fq = pd.concat([old_df_fq, new_df_fq], ignore_index=True).set_index('cod_sampler')

#     idx = df_sens.index.intersection(df_fq.index)
#     df_fq = df_fq[idx]
#     df_sens = df_sens[idx]

#     model = models.model_s2fq()
#     model.fit(df_sens.values/10, df_fq.values)
#     model.save("")
#     return ""

# @app.post("/retrain_model_sens")
# async def retrain_model_sens():
#     old_df_fq = read_from_db(OLD_DATA, 'train_fq')
#     new_df_fq = read_from_db(NEW_DATA, 'train_fq')
#     df_fq = pd.concat([old_df_fq, new_df_fq], ignore_index=True).set_index('cod_sampler')
#     for var in SENS_VARS_CHOC:
#         old_df_var = read_from_db(OLD_DATA, f'train_{var}')
#         new_df_var = read_from_db(NEW_DATA, f'train_{var}')
#         df_var = pd.concat([old_df_fq, new_df_fq], ignore_index=True).set_index('cod_sampler')
#         idx = df_var.index.intersection(df_fq.index)
#         y = df_var[idx].values.round()
#         X = df_fq[idx].values
#         iAnn = get_iAnn(y)
#         y = np.nan_to_num(y)
#         model = MA_GCCE(R=len(ANOTADORES_CHOC), K=10, learning_rate=1e-4, verbose=0)
#         model.fit(X, y)
#         models.save("")
#     return ""