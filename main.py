from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import numpy as np
import tensorflow as tf
import pickle

from src import models
from src.multiple_annotators_models import MA_GCCE
from src.utils import get_iAnn, transform_data
from src.parameters import *

app = FastAPI()

# Dependency to load models
def get_model_sens():
    return app.state.model_to_sens

def get_model_fq():
    return app.state.model_to_fq

def get_scaler_fq():
    return app.state.scaler_fq

# Load models (moved to app startup event)
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

    with open('models\gcce\scaler_X.pkl', 'rb') as file:
        scaler_X = pickle.load(file)
    app.state.scaler_fq = scaler_X
# Definir constante global para el número mínimo de muestras
MIN_SAMPLES = 10

# Schemas
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

    # Validator to ensure minimum number of samples
    @validator("data")
    def check_min_samples(cls, v):
        if len(v) < MIN_SAMPLES:  # Accede a la constante global en vez de un atributo de clase
            raise ValueError(f"Se necesitan al menos {MIN_SAMPLES} muestras para el entrenamiento.")
        return v

class TrainingItem2(BaseModel):
    physicochemical_data: PhysicochemicalData
    sensory_data_by_user: Dict[str, UserSensoryData]

class TrainingData2(BaseModel):
    family: str
    type_model: int
    data: List[TrainingItem2]

    # Validator to ensure minimum number of samples
    @validator("data")
    def check_min_samples(cls, v):
        if len(v) < MIN_SAMPLES:  # Accede a la constante global en vez de un atributo de clase
            raise ValueError(f"Se necesitan al menos {MIN_SAMPLES} muestras para el entrenamiento.")
        return v

# Prediction endpoints using injected dependencies
@app.post("/predict_sens")
async def predict_sens(data: FqInput, model_to_sens=Depends(get_model_sens), scaler_fq=Depends(get_scaler_fq)):
    input_data = np.array([[data.humidity_halogen, data.fat_nmr, data.granulometry_micrometer,
                            data.plastic_viscosity_anton_paar, data.flow_limit_anton_paar]])
    input_data = scaler_fq.transform(input_data)
    predictions, _ = model_to_sens.predict(input_data)

    pred_values = predictions.flatten().tolist()
    return {TRANS_SENS_VARS_CHOC[SENS_VARS_CHOC[i]]: pred_values[i] for i in range(len(pred_values))}

@app.post("/predict_fq")
async def predict_fq(data: SensInput, model_to_fq=Depends(get_model_fq), scaler_fq=Depends(get_scaler_fq)):
    input_data = np.array([[data.acidity, data.bitterness, data.aroma, data.astringency,
                            data.sweetness, data.hardness, data.global_impression, data.melting_speed]])
    input_data /= 10
    predictions = model_to_fq.predict(input_data)
    predictions = scaler_fq.inverse_transform(predictions)
    pred_values = predictions.flatten().tolist()
    return {TRANS_FQ_VARS_CHOC[FQ_VARS_CHOC[i]]: pred_values[i] for i in range(len(pred_values))}

# Retraining endpoint with validation for sensory model
@app.post("/retrain_model_sens")
async def retrain_model_sens(training_data: TrainingData2, model_to_sens=Depends(get_model_sens), scaler_fq=Depends(get_scaler_fq)):
    # Check minimum number of samples
    if len(training_data.data) < MIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Se necesitan al menos {MIN_SAMPLES} muestras para el entrenamiento.")
    # Transform and retrain the model
    Y, X = transform_data(training_data)
    X = scaler_fq.transform(X)
    for var in SENS_VARS_CHOC:
        y = np.nan_to_num(np.round(Y[TRANS_SENS_VARS_CHOC[var]]))
        model = MA_GCCE(R=y.shape[1], K=10, learning_rate=1e-4, verbose=0)
        model.fit(X, y)
        model.model.save(f"models/gcce/gcce_ma_{var}.keras")

    return {"message": "Modelo entrenado exitosamente"}

# Retraining endpoint with validation for physicochemical model
@app.post("/retrain_model_fq")
async def retrain_model_fq(training_data: TrainingData, model_to_fq=Depends(get_model_fq), scaler_fq=Depends(get_scaler_fq)):
    # Check minimum number of samples
    if len(training_data.data) < MIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Se necesitan al menos {MIN_SAMPLES} muestras para el entrenamiento.")

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
    Y = scaler_fq.transform(Y)
    # Retrain the model with new data
    model_to_fq.fit(X/10, Y)
    model_to_fq.save("models/gcce/model_s2fq.keras")
    return {"message": "Modelo entrenado exitosamente"}
