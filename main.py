from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import tensorflow as tf
# import sqlite3
import pandas as pd

from src import models
from src.multiple_annotators_models import MA_GCCE
from src.utils import get_iAnn, transform_data
from src.parameters import *
from fastapi import FastAPI, Depends
import tensorflow as tf
import numpy as np

app = FastAPI()

# %% Dependency to load models
def get_model_sens():
    return app.state.model_to_sens

def get_model_fq():
    return app.state.model_to_fq

# %% load models (moved to app startup event)
@app.on_event("startup")
def load_models():
    def GCCE_MA_loss(y_true, y_pred):
        return MA_GCCE.GCCE_MA_loss(y_true, y_pred)

    # Load sensory model
    models_sens = []
    for var in SENS_VARS_CHOC:
        models_sens.append(tf.keras.models.load_model(f"models/gcce/gcce_ma_{var}.keras",
                                                      custom_objects={'method': GCCE_MA_loss}))
    app.state.model_to_sens = models.MultiOutputModel(models_sens)

    # Load physicochemical model
    app.state.model_to_fq = tf.keras.models.load_model('models/gcce/model_s2fq.keras')

# %% schemas
class FqInput(BaseModel):
    humidity_halogen : float
    fat_nmr : float
    granulometry_micrometer : float
    plastic_viscosity_anton_paar : float
    flow_limit_anton_paar : float
    reference_id : str

class SensInput(BaseModel):
    acidity : float
    bitterness : float
    aroma : float
    astringency : float
    sweetness : float
    hardness : float
    global_impression : float
    melting_speed : float
    reference_id : str

class SensoryData(BaseModel):
    acidity: Optional[float]
    bitterness: Optional[float]
    aroma: Optional[float]
    astringency: Optional[float]
    sweetness: Optional[float]
    hardness: Optional[float]
    global_impression: Optional[float]
    melting_speed: Optional[float]

class UserSensoryData(BaseModel):
    sensory_data: List[SensoryData]

class PhysicochemicalData(BaseModel):
    humidity_halogen: float
    fat_nmr: float
    granulometry_micrometer: float
    plastic_viscosity_anton_paar: float
    flow_limit_anton_paar: float

class TrainingItem(BaseModel):
    sensory_data: SensoryData
    physicochemical_data: PhysicochemicalData

class TrainingData(BaseModel):
    family: str
    type_model: int
    data: List[TrainingItem]

class TrainingItem2(BaseModel):
    physicochemical_data: PhysicochemicalData
    sensory_data_by_user: Dict[str, UserSensoryData]

    # Define the main schema for the training data
class TrainingData2(BaseModel):
    family: str
    type_model: int
    data: List[TrainingItem2]

# %% prediction endpoints using injected dependencies
@app.post("/predict_sens")
async def predict_sens(data: FqInput, model_to_sens=Depends(get_model_sens)):
    input_data = np.array([[data.humidity_halogen, data.fat_nmr, data.granulometry_micrometer,
                            data.plastic_viscosity_anton_paar, data.flow_limit_anton_paar]])
    
    predictions, _ = model_to_sens.predict(input_data)

    pred_values = predictions.flatten().tolist()
    return {TRANS_SENS_VARS_CHOC[SENS_VARS_CHOC[i]]: pred_values[i] for i in range(len(pred_values))}

@app.post("/predict_fq")
async def predict_fq(data: SensInput, model_to_fq=Depends(get_model_fq)):
    input_data = np.array([[data.acidity, data.bitterness, data.aroma, data.astringency,
                            data.sweetness, data.hardness, data.global_impression, data.melting_speed]])
    predictions = model_to_fq.predict(input_data)

    pred_values = predictions.flatten().tolist()
    return {TRANS_FQ_VARS_CHOC[FQ_VARS_CHOC[i]]: pred_values[i] for i in range(len(pred_values))}

# %% retraining endpoint
@app.post("/retrain_model_fq")
async def retrain_model_fq(training_data: TrainingData, model_to_fq=Depends(get_model_fq)):
    sensory_data_list = []
    physicochemical_data_list = []

    for item in training_data.data:
        # Convert sensory_data to a list of feature values (X)
        sensory_data_list.append([
            item.sensory_data.acidity,
            item.sensory_data.bitterness,
            item.sensory_data.aroma,
            item.sensory_data.astringency,
            item.sensory_data.sweetness,
            item.sensory_data.hardness,
            item.sensory_data.global_impression,
            item.sensory_data.melting_speed
        ])

        # Convert physicochemical_data to a list of target values (Y)
        physicochemical_data_list.append([
            item.physicochemical_data.humidity_halogen,
            item.physicochemical_data.fat_nmr,
            item.physicochemical_data.granulometry_micrometer,
            item.physicochemical_data.plastic_viscosity_anton_paar,
            item.physicochemical_data.flow_limit_anton_paar
        ])

    # Convert lists to NumPy arrays
    X = np.array(sensory_data_list)
    Y = np.array(physicochemical_data_list)

    # Retrain the model with new data
    model_to_fq.fit(X, Y)
    model_to_fq.save("models/gcce/model_s2fq.keras")
    return {"message": "Modelo entrenado exitosamente"}

@app.post("/retrain_model_sens")
async def retrain_model_sens(training_data: TrainingData2, model_to_sens=Depends(get_model_sens)):
    Y, X = transform_data(training_data)
    for var in SENS_VARS_CHOC:
        y = np.nan_to_num(np.round(Y[TRANS_SENS_VARS_CHOC[var]]))
        model = MA_GCCE(R=y.shape[1], K=10, learning_rate=1e-4, verbose=0)
        model.fit(X, y)
        model.model.save(f"models/gcce/gcce_ma_{var}.keras")
    return {"message": "Modelo entrenado exitosamente"}