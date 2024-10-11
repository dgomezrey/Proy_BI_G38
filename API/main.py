# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
from Proyecto1Parte2 import TextPreprocessor


# Cargar el modelo entrenado
pipeline = load('modelo_entrenado_completo.joblib')

# Definir la aplicación FastAPI
app = FastAPI()

# Definir la clase de datos de entrada
class TextData(BaseModel):
    textos: list[str]

# Endpoint 1: Predicción
@app.post("/predict")
def predict(data: TextData):
    # Convertir los textos recibidos en un DataFrame
    input_data = pd.DataFrame(data.textos, columns=['Textos_espanol'])
    
    # Realizar la predicción con el modelo cargado
    predicciones = pipeline.predict(input_data['Textos_espanol'])
    probabilidades = pipeline.predict_proba(input_data['Textos_espanol'])
    
    # Preparar las respuestas con las predicciones y probabilidades
    resultados = [{"prediccion": int(pred), "probabilidades": prob.tolist()} for pred, prob in zip(predicciones, probabilidades)]
    
    return {"resultados": resultados}

# Definir la clase para datos de reentrenamiento
class RetrainData(BaseModel):
    textos: list[str]
    etiquetas: list[int]

# Endpoint 2: Reentrenamiento
@app.post("/retrain")
def retrain(data: RetrainData):
    # Convertir los datos recibidos en un DataFrame
    input_data = pd.DataFrame(data.textos, columns=['Textos_espanol'])
    etiquetas = pd.Series(data.etiquetas)
    
    # Reentrenar el modelo con los nuevos datos
    pipeline.fit(input_data['Textos_espanol'], etiquetas)
    
    # Guardar el modelo actualizado
    load.dump(pipeline, 'modelo_entrenado_completo.joblib')
    
    # Calcular métricas de desempeño
    y_pred = pipeline.predict(input_data['Textos_espanol'])
    precision = precision_score(etiquetas, y_pred, average='macro')
    recall = recall_score(etiquetas, y_pred, average='macro')
    f1 = f1_score(etiquetas, y_pred, average='macro')
    
    return {"precision": precision, "recall": recall, "f1_score": f1}
