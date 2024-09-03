from fastapi import FastAPI, Query
from pydantic import BaseModel
from managedb import SQLiteDB
import tensorflow as tf
from src import models
from src.multiple_annotators_models import MA_GCCE
import numpy as np
from src.parameters import SENS_VARS_CHOC
import uvicorn
import json
#import datetime

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

@app.get("/predict_sens")
async def predict_sens(
    humedad: float = Query(...),
    grasa: float = Query(...),
    granulometria: float = Query(...),
    viscosidad: float = Query(...),
    fluidez: float = Query(...)
):
    
    input_data = np.array([[humedad, grasa, granulometria, viscosidad, fluidez]])
    predictions = model_to_sens.predict(input_data)
    # Guardar la predicción en la base de datos
    db = SQLiteDB("predictions.db")
    sens_data = {
        "humedad": humedad,
        "grasa": grasa,
        "granulometria": granulometria,
        "viscosidad": viscosidad,
        "fluidez": fluidez,
        "result": json.dumps(predictions.tolist()),  # Convertir la predicción a JSON
        #"created_at": datetime.now().isoformat()  # Guardar la fecha y hora actual
    }
    db.insert_data("SensPrediction", sens_data)
    db.close()
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
    # Predecir con el modelo
    input_data = np.array([[acidez, amargor, aroma, astringencia, dulce, dureza, impresion, fusion]])
    predictions = model_to_fq.predict(input_data)
    
    # Guardar la predicción en la base de datos
    db = SQLiteDB("predictions.db")
    fq_data = {
        "acidez": acidez,
        "amargor": amargor,
        "aroma": aroma,
        "astringencia": astringencia,
        "dulce": dulce,
        "dureza": dureza,
        "impresion": impresion,
        "fusion": fusion,
        "result": json.dumps(predictions.tolist()),  # Convertir la predicción a JSON
        #"created_at": datetime.now().isoformat()  # Guardar la fecha y hora actual
    }
    db.insert_data("FqPrediction", fq_data)
    db.close()
    
    return {"fq": predictions.tolist()}


@app.get("/create_db/")
async def createdb():
    # Crear y usar la tabla Prediction
    db = SQLiteDB("predictions.db")

    # Definición de las columnas para SensPrediction
    sens_columns = {
        "humedad": "REAL",
        "grasa": "REAL",
        "granulometria": "REAL",
        "viscosidad": "REAL",
        "fluidez": "REAL",
        "result": "TEXT",  # Guardado como JSON en texto
        "created_at": "TEXT"
    }
    db.create_table("SensPrediction", sens_columns)

    # Definición de las columnas para FqPrediction
    fq_columns = {
        "acidez": "REAL",
        "amargor": "REAL",
        "aroma": "REAL",
        "astringencia": "REAL",
        "dulce": "REAL",
        "dureza": "REAL",
        "impresion": "REAL",
        "fusion": "REAL",
        "result": "TEXT",  # Guardado como JSON en texto
        "created_at": "TEXT"
    }
    db.create_table("FqPrediction", fq_columns)

    db.close()
    return {"SQL": True}

@app.get("/retrain_model/")
async def retrain_model(familia: str = Query(...)):
    db = SQLiteDB("predictions.db")
    
    if familia == "sens":
        X, y = load_data_from_db(db, "SensPrediction")
        model = model_to_sens  # Cargar el modelo adecuado
    elif familia == "fq":
        X, y = load_data_from_db(db, "FqPrediction")
        model = model_to_fq  # Cargar el modelo adecuado
    else:
        return {"error": "Invalid family type"}
    
    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = split_data.split_data(X, y)
    
    # Normalizar los datos de entrenamiento y validación
    X_train, X_val = normalize_data.normalize_data(X_train, X_val)
    
    # Reentrenar el modelo con los datos normalizados
    retrained_model_history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    
    # Guardar el modelo reentrenado
    save_retrained_model.save_retrained_model(model, "models/gcce/model_s2fq_retrained.keras")
    
    db.close()
    
    return {"retrain": "success"}



def load_data_from_db(db, table_name):
    raw_data = db.read_data(table_name)
    inputs = []
    outputs = []
    
    for row in raw_data:
        if table_name == "FqPrediction":
            input_data = [
                row[0],  # 'acidez'
                row[1],  # 'amargor'
                row[2],  # 'aroma'
                row[3],  # 'astringencia'
                row[4],  # 'dulce'
                row[5],  # 'dureza'
                row[6],  # 'impresion'
                row[7]   # 'fusion'
            ]
        output_data = json.loads(row[8])  # 'result' debería estar en la posición 8
        
        inputs.append(input_data)
        outputs.append(output_data)
    
    return np.array(inputs), np.array(outputs)




def split_data(X, y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)

def normalize_data(X_train, X_val):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

def save_retrained_model(model, path):
    model.save(path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
