import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocess_and_split(df: pd.DataFrame, target: str):
    """
    Procesa el dataframe y prepara los datos para el pipeline de imblearn.
    
    Returns:
        X (pd.DataFrame): DataFrame con las características procesadas.
        y (pd.Series): Serie con la variable objetivo.
        preprocessor (ColumnTransformer): El objeto preprocesador ajustado.
    """
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    df[target] = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

    categorical_features = df.select_dtypes(include=['object']).columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(target)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # En esta versión, solo devolvemos el DataFrame completo 
    # El pipeline de GridSearchCV manejará el split y el entrenamiento.
    X = df.drop(columns=[target], axis=1)
    y = df[target]

    return X, y, preprocessor