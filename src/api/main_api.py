# src/api/main_api.py
# API para predecir el abandono de clientes utilizando FastAPI y un modelo preentrenado
import joblib
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import numpy as np  # <-- Added this import

# Importar las funciones de preprocesamiento del proyecto
from src.features.feature_engineering import create_features, consolidate_categories

# Crear la aplicación FastAPI
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para predecir el abandono de clientes de telecomunicaciones.",
    version="1.0.0"
)

# Definir la ruta del modelo global
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best_overall_model.pkl')

# Cargar el modelo al inicio del servidor
try:
    best_overall_model = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado exitosamente.")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo del modelo en {MODEL_PATH}.")
    best_overall_model = None
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    best_overall_model = None

# Definir el esquema de los datos de entrada con Pydantic
class ChurnPredictionRequest(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float
    TotalCharges: float
    customerID: str 

@app.get("/")
def read_root():
    return {"message": "API de predicción de Churn está operativa."}


# Definir el endpoint de predicción
@app.post("/predict")
def predict_churn(request_data: ChurnPredictionRequest):
    if best_overall_model is None:
        return {"error": "El modelo no se ha cargado correctamente."}

    # Convertir los datos de la solicitud a un DataFrame
    data_dict = request_data.dict()
    df = pd.DataFrame([data_dict])
    
    # SE AÑADE LA GESTIÓN DE COLUMNA customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Aplicar las mismas transformaciones de ingeniería de características
    df = consolidate_categories(df)
    df = create_features(df)
    
    # Realizar la predicción
    prediction = best_overall_model.predict(df)[0]
    
    # Obtener las probabilidades de predicción
    prediction_proba = best_overall_model.predict_proba(df)[0]

    # Devolver la predicción
    return {
        "prediccion_churn": "Sí" if prediction == 1 else "No",
        "probabilidad_no_churn": round(float(prediction_proba[0]), 4),  # <-- Se convierte a float
        "probabilidad_churn": round(float(prediction_proba[1]), 4),    # <-- Se convierte a float
        "mensaje": "Predicción realizada exitosamente."
    }