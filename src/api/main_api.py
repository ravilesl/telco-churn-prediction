from fastapi import FastAPI
from pydantic import BaseModel
from src.api.predict_model import ChurnPredictor
import os

# --- Definición de la API ---
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para predecir si un cliente de telecomunicaciones abandonará el servicio."
)

# --- Esquema de Entrada ---
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# --- Cargar el modelo ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'churn_model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'preprocessor.pkl')

try:
    predictor = ChurnPredictor(MODEL_PATH, PREPROCESSOR_PATH)
except FileNotFoundError:
    print("Modelo o preprocesador no encontrados. Por favor, ejecute el pipeline de entrenamiento.")
    predictor = None

# --- Rutas de la API ---
@app.get("/")
def home():
    return {"message": "Bienvenido a la API de Predicción de Churn."}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    """Ruta para realizar una predicción de churn."""
    if predictor is None:
        return {"error": "El modelo no ha sido entrenado. Por favor, entrene el modelo primero."}
        
    prediction = predictor.predict_single(customer.dict())
    return {"prediction": prediction, "result": "Churn" if prediction == 1 else "No Churn"}