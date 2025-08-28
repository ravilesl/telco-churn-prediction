# src/api/predict_model.py 
# Módulo para cargar el modelo y realizar predicciones
import joblib
import pandas as pd
from typing import Dict, Any

class ChurnPredictor:
    """Clase para cargar y predecir con el modelo entrenado."""

    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict_single(self, data: Dict[str, Any]) -> int:
        """
        Realiza una predicción sobre los datos de entrada.
        Retorna 1 si hay churn, 0 si no.
        """
        # Creación de DataFrame con las columnas del preprocesador
        df = pd.DataFrame([data])
        X_processed = self.preprocessor.transform(df)
        prediction = self.model.predict(X_processed)
        return int(prediction[0])