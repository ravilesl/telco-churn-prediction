import joblib
import pandas as pd
from src.utils.data_loader import load_raw_data
from src.features.feature_engineering import preprocess_and_split
from src.models.model_factory import ModelFactory
from src.evaluation.metrics_evaluator import evaluate_model
import os
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

# Umbral para considerar un dataset desbalanceado
IMBALANCE_THRESHOLD = 0.10  # 10% de la clase minoritaria

def detect_class_imbalance(y):
    """
    Detecta si el dataset está desbalanceado y retorna la métrica de scoring 
    y los parámetros de búsqueda iniciales.
    """
    class_counts = pd.Series(y).value_counts(normalize=True)
    minority_class_ratio = class_counts.min()
    print(f"\nProporción de la clase minoritaria: {minority_class_ratio:.2%}")

    scoring_metric = 'accuracy'
    initial_params_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
    }

    if minority_class_ratio < IMBALANCE_THRESHOLD:
        print("Dataset desbalanceado. Ajustando parámetros iniciales para optimizar la clase minoritaria.")
        scoring_metric = 'f1' # Usamos F1-Score
        scale_pos_weight_value = class_counts[0] / class_counts[1]
        
        initial_params_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [5, 7],
            # `scale_pos_weight` solo es relevante si no se usa sobremuestreo
            'model__scale_pos_weight': [scale_pos_weight_value, 1],
        }
    else:
        print("Dataset balanceado. Usando parámetros iniciales estándar.")
        
    return scoring_metric, initial_params_grid

def run_training_pipeline(data_path: str):
    """Orquesta el pipeline completo de entrenamiento y evaluación."""
    
    df = load_raw_data(data_path)
    if df is None:
        return

    # Dividimos los datos para la evaluación final
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # 1. Preprocesar los datos de entrenamiento para la búsqueda
    X_train, y_train, preprocessor = preprocess_and_split(df_train, 'Churn')
    print("Datos de entrenamiento preprocesados y listos.")
    
    # 2. Detectar el desbalance de clases y seleccionar la métrica y los parámetros
    scoring_metric, initial_params_grid = detect_class_imbalance(y_train)

    # 3. Definir las técnicas de balanceo a probar
    balancing_techniques = {
        'no_balancing': None,
        'smote': SMOTE(random_state=42),
        'adasyn': ADASYN(random_state=42),
        'random_oversampler': RandomOverSampler(random_state=42), # Nuevo
        'random_undersampler': RandomUnderSampler(random_state=42) # Nuevo
    }

    best_overall_score = 0
    best_overall_model = None
    best_overall_technique = ''
    
    # 4. Iterar a través de cada técnica de balanceo y encontrar el mejor modelo
    for tech_name, sampler in balancing_techniques.items():
        print(f"\n--- Probando la técnica: {tech_name} ---")

        # 5. Crear el pipeline completo (opcional de balanceo y modelo)
        steps = [('preprocessor', preprocessor)]
        if sampler:
            steps.append(('sampler', sampler))
            
        steps.append(('model', ModelFactory.create_model_base('xgboost')))
        model_pipeline = Pipeline(steps=steps)

        # 6. Crear el GridSearchCV con el pipeline dinámico
        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=initial_params_grid,
            cv=5,
            scoring=scoring_metric,
            n_jobs=-1,
            verbose=1
        )
        
        # 7. Entrenar el modelo con la búsqueda de hiperparámetros
        grid_search.fit(X_train, y_train)
        
        print(f"\nMejor {scoring_metric} para {tech_name}: {grid_search.best_score_:.4f}")
        print(f"Mejores parámetros: {grid_search.best_params_}")

        # 8. Comparar y guardar el mejor modelo global
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_model = grid_search.best_estimator_
            best_overall_technique = tech_name

    print("\n=======================================================")
    print(f"Mejor modelo global encontrado con la técnica: {best_overall_technique}")
    print(f"Mejor {scoring_metric} global: {best_overall_score:.4f}")
    print("=======================================================")

    # 9. Preprocesar los datos de prueba usando el mismo preprocesador
    X_test, y_test, _ = preprocess_and_split(df_test, 'Churn')
    
    # 10. Realizar predicciones y evaluar con el mejor modelo global
    y_pred = best_overall_model.predict(X_test)
    evaluate_model(y_test, y_pred, 'XGBoost (Optimizado)')

    # 11. Crear el directorio 'models' si no existe
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # 12. Guardar el mejor modelo global y el preprocesador
    joblib.dump(best_overall_model, os.path.join(models_dir, 'best_overall_model.pkl'))
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
    
    print("\nMejor modelo y preprocesador guardados en la carpeta 'models'.")
    print("\nPipeline de entrenamiento completado.")