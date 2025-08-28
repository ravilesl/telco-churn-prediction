# src/features/feature_engineering.py
# Módulo para el procesamiento de datos y la ingeniería de características
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import List, Tuple

def consolidate_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida las categorías 'No internet service' y 'No phone service' a 'No'
    en las columnas categóricas relevantes.
    """
    df_copy = df.copy()
    columns_to_consolidate = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    for col in columns_to_consolidate:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(['No internet service', 'No phone service'], 'No')
    return df_copy

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas características para mejorar el rendimiento del modelo.
    """
    df_copy = df.copy()
    
    # 1. Conteo de servicios de internet y TV/Música
    internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_copy['InternetServiceCount'] = df_copy[internet_services].apply(lambda row: sum(1 for item in row if item == 'Yes'), axis=1)

    # 2. Característica de interacción: permanencia vs. contrato
    df_copy['tenure_contract_interaction'] = df_copy['tenure'] * df_copy['Contract'].apply(
        lambda x: 24 if x == 'Two year' else (12 if x == 'One year' else 1)
    )

    return df_copy

def preprocess_and_split(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Procesa el dataframe y prepara los datos para el pipeline de imblearn.
    
    Returns:
        X (pd.DataFrame): DataFrame con las características procesadas.
        y (pd.Series): Serie con la variable objetivo.
        preprocessor (ColumnTransformer): El objeto preprocesador ajustado.
    """
    # Excluir explícitamente customerID al inicio del preprocesamiento
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Manejar los valores nulos en TotalCharges antes de la conversión
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    
    # Consolidar categorías para mejorar la codificación
    df = consolidate_categories(df)
    
    # Crear nuevas características
    df = create_features(df)
    
    df[target] = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

    # Separar características y variable objetivo
    X = df.drop(columns=[target], axis=1)
    y = df[target]

    # Identificar las características numéricas y categóricas
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Crear el preprocesador para las columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    return X, y, preprocessor