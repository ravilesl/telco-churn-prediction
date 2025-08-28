# src/api/main_api.py
# API para predecir el abandono de clientes utilizando FastAPI y un modelo preentrenado
import joblib
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

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
    customerID: str # Se agrega para manejarlo en la API

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Abandono de Clientes",
    description="Una API simple para predecir si un cliente de Telco abandonará o no."
)

# Definir el endpoint de predicción
@app.post("/predict")
def predict_churn(request_data: ChurnPredictionRequest):
    if best_overall_model is None:
        return {"error": "El modelo no se ha cargado correctamente."}

    # Convertir los datos de la solicitud a un DataFrame
    input_df = pd.DataFrame([request_data.dict()])

    # El preprocesador es parte del pipeline y manejará la conversión.
    # El pipeline completo de GridSearchCV que se guardó se encargará de esto.
    
    prediction = best_overall_model.predict(input_df)[0]
    
    # Obtener las probabilidades de predicción
    prediction_proba = best_overall_model.predict_proba(input_df)

    # Devolver la predicción
    return {
        "prediccion_churn": "Sí" if prediction == 1 else "No",
        "probabilidad_no_churn": round(prediction_proba[0][0], 4),
        "probabilidad_churn": round(prediction_proba[0][1], 4),
        "mensaje": "Predicción realizada exitosamente."
    }