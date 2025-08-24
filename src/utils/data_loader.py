import pandas as pd
from typing import Optional

def load_raw_data(file_path: str) -> Optional[pd.DataFrame]:
    """Carga los datos crudos desde un archivo CSV. Adhiere al SRP."""
    try:
        df = pd.read_csv(file_path)
        print("Datos cargados exitosamente.")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo no fue encontrado en {file_path}")
        return None
    

if __name__ == "__main__":
    # Ruta temporal para la prueba
    # Descarga el archivo si no lo tienes
    import os
    if not os.path.exists('../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
        print("Descargando dataset...")
        os.system('mkdir -p ../../data/raw && wget -O ../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-extra-data/master/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    df = load_raw_data('../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if df is not None:
        print("Datos cargados correctamente. Filas:", df.shape[0])
        print("Columnas:", df.columns)
    else:
        print("Fallo al cargar los datos.")