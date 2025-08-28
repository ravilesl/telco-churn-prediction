# src/pipelines/training_pipeline.py
# Pipeline completo de entrenamiento, evaluación y guardado del modelo
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from typing import Dict, Any

from src.utils.data_loader import load_raw_data
from src.features.feature_engineering import preprocess_and_split
from src.models.model_factory import ModelFactory
from src.evaluation.metrics_evaluator import evaluate_model
from src.models.model_factory import ModelFactory

# Umbral para considerar un dataset desbalanceado
HIGH_IMBALANCE_THRESHOLD = 0.10
SEMI_IMBALANCE_THRESHOLD = 0.30

def detect_class_imbalance(y):
    """
    Detecta si el dataset está desbalanceado y retorna la métrica de scoring,
    la grilla de parámetros iniciales y las técnicas de balanceo a probar.
    """
    class_counts = pd.Series(y).value_counts(normalize=True)
    minority_class_ratio = class_counts.min()
    print(f"\nProporción de la clase minoritaria: {minority_class_ratio:.2%}")

    balancing_techniques = {'no_balancing': None}
    scoring_metric = 'accuracy'

    if minority_class_ratio < HIGH_IMBALANCE_THRESHOLD:
        print("Dataset altamente desbalanceado. Usando f1-score y técnicas de sobremuestreo agresivas.")
        scoring_metric = 'f1'
        balancing_techniques.update({
            'smote': SMOTE(random_state=42),
            'adasyn': ADASYN(random_state=42)
        })
    elif HIGH_IMBALANCE_THRESHOLD <= minority_class_ratio < SEMI_IMBALANCE_THRESHOLD:
        print("Dataset semidesbalanceado. Usando f1-score y técnicas de muestreo aleatorio.")
        scoring_metric = 'f1'
        balancing_techniques.update({
            'random_oversampler': RandomOverSampler(random_state=42),
            'random_undersampler': RandomUnderSampler(random_state=42)
        })
    else:
        print("Dataset balanceado. Usando accuracy y no se aplican técnicas de balanceo.")

    return scoring_metric, balancing_techniques

def run_training_pipeline(data_path: str):
    """Orquesta el pipeline completo de entrenamiento y evaluación."""
    
    df = load_raw_data(data_path)
    if df is None:
        return

    # Dividimos los datos para la evaluación final
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Preprocesar los datos de entrenamiento para la búsqueda
    X_train, y_train, preprocessor = preprocess_and_split(df_train, 'Churn')
    print("Datos de entrenamiento preprocesados y listos.")
    
    scoring_metric, balancing_techniques = detect_class_imbalance(y_train)

    best_overall_score = 0
    best_overall_model = None
    best_overall_technique = ''
    best_overall_model_name = ''
    
    models_to_test = [
        'xgboost', 'lightgbm', 'gradient_boosting', 'random_forest',
        'adaboost', 'logistic_regression', 'decision_tree', 'knn', 'svc'
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n=======================================================")
        print(f"Probando el modelo: {model_name.upper()}")
        
        for tech_name, sampler in balancing_techniques.items():
            print(f"--- Probando la técnica: {tech_name} ---")

            model_base, initial_params_grid = ModelFactory.create_model_and_params(model_name)
            
            # Ajustar los parámetros para manejar el desbalance de clases si no se usa sobremuestreo
            if tech_name == 'no_balancing':
                class_counts = pd.Series(y_train).value_counts(normalize=True)
                scale_pos_weight_value = class_counts[0] / class_counts[1]
                # Asegúrate de que los modelos que no tienen este parámetro no se vean afectados
                if model_name in ['xgboost', 'lightgbm']:
                    initial_params_grid['model__scale_pos_weight'] = [scale_pos_weight_value, 1]
                elif model_name in ['logistic_regression', 'svc']:
                    initial_params_grid['model__class_weight'] = [None, 'balanced']


            steps = [('preprocessor', preprocessor)]
            if sampler:
                steps.append(('sampler', sampler))
            
            steps.append(('model', model_base))
            model_pipeline = Pipeline(steps=steps)

            grid_search = GridSearchCV(
                estimator=model_pipeline,
                param_grid=initial_params_grid,
                cv=5,
                scoring=scoring_metric,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"\nMejor {scoring_metric} para {tech_name}: {grid_search.best_score_:.4f}")
            print(f"Mejores parámetros: {grid_search.best_params_}")

            current_score = grid_search.best_score_
            key = f"{model_name}_{tech_name}"
            results[key] = current_score

            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_model = grid_search.best_estimator_
                best_overall_technique = tech_name
                best_overall_model_name = model_name

    print("\n=======================================================")
    print(f"Mejor modelo global encontrado: {best_overall_model_name.upper()} con la técnica: {best_overall_technique}")
    print(f"Mejor {scoring_metric} global: {best_overall_score:.4f}")
    print("=======================================================")

    plt.figure(figsize=(15, 8))
    results_df = pd.Series(results).reset_index()
    results_df.columns = ['Model_Technique', 'Score']
    sns.barplot(data=results_df, x='Score', y='Model_Technique', palette='viridis')
    plt.title('Comparación de Modelos y Técnicas de Balanceo (F1-Score)')
    plt.xlabel('F1-Score')
    plt.ylabel('Modelo y Técnica')
    plt.tight_layout()
    plt.savefig('assets/model_comparison.png')
    plt.show()

    X_test, y_test, _ = preprocess_and_split(df_test, 'Churn')
    y_pred = best_overall_model.predict(X_test)
    evaluate_model(y_test, y_pred, f'{best_overall_model_name.upper()} (Optimizado)')

    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    joblib.dump(best_overall_model, os.path.join(models_dir, 'best_overall_model.pkl'))
    
    print("\nMejor modelo guardado en la carpeta 'models'.")
    print("\nPipeline de entrenamiento completado.")