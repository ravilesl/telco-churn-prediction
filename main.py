# main.py
# Punto de entrada principal para ejecutar el pipeline de entrenamiento
import os
from src.pipelines.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    run_training_pipeline(DATA_PATH)