# src/evaluation/metrics_evaluator.py
# Módulo para evaluar el rendimiento del modelo y generar reportes
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(y_true, y_pred, model_name: str):
    """Evalúa las predicciones del modelo y genera un reporte."""
    report = classification_report(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- Reporte de Evaluación para {model_name} ---")
    print(report)
    print(f"F1-Score: {f1:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'Matriz de Confusión para {model_name}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig('assets/confusion_matrix.png')
    plt.close()

    return report, f1, cm