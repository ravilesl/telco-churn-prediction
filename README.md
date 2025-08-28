# Predicción de Abandono de Clientes (Telco Churn Prediction)

Este proyecto implementa un pipeline de **aprendizaje automático** para predecir la tasa de abandono de clientes de una compañía de telecomunicaciones. El objetivo principal es identificar a los clientes en riesgo de irse (`Churn='Yes'`) para que la empresa pueda tomar medidas de retención proactivas.

El proyecto está diseñado con una arquitectura modular y escalable, siguiendo las mejores prácticas de **Ingeniería de MLOps**, lo que facilita su mantenimiento, depuración y la adición de nuevos modelos o características.

## Arquitectura del Proyecto

La estructura del repositorio está organizada para separar las distintas etapas del pipeline de ML:

- **`main.py`**: El script principal que orquesta todo el flujo de trabajo, desde la carga de datos hasta el entrenamiento y la evaluación de modelos.
- **`src/`**: Directorio principal que contiene el código fuente del proyecto.
  - **`data/`**: Contiene la lógica para la carga y gestión de los datos.
  - **`features/`**: Módulo para el preprocesamiento de datos y la ingeniería de características.
  - **`models/`**: Contiene las definiciones de los modelos y la lógica para el ajuste de hiperparámetros.
  - **`pipelines/`**: Módulo que define y ejecuta el pipeline de entrenamiento completo.
  - **`utils/`**: Funciones de utilidad y herramientas de ayuda.
- **`notebooks/`**: Cuadernos de Jupyter/Colab para la exploración de datos y la demostración del pipeline.
- **`models/`**: Carpeta para guardar el modelo entrenado y serializado.
- **`assets/`**: Almacena los gráficos y las visualizaciones de la evaluación del modelo.

## Archivos y Métodos Clave

A continuación, se describen los componentes principales del proyecto:

### 📁 `src/data/data_loader.py`

- **`load_raw_data(file_path)`**: Función que se encarga de cargar el archivo CSV del dataset.

### 📁 `src/features/feature_engineering.py`

Este módulo es fundamental para preparar los datos. Implementa varias estrategias de preprocesamiento y enriquecimiento:

- **`consolidate_categories(df)`**: Transforma valores como **"No internet service"** y **"No phone service"** a un valor unificado de **"No"**. Esta estrategia simplifica las categorías, reduciendo la dimensionalidad del dataset y ayudando a los modelos a generalizar mejor.
- **`create_features(df)`**: Aplica **ingeniería de características** para generar nuevas variables a partir de las existentes, como:
  - **`InternetServiceCount`**: Un conteo de servicios de internet que el cliente tiene contratados.
  - **`tenure_contract_interaction`**: Una característica de interacción que combina el tiempo de permanencia (`tenure`) con el tipo de contrato, capturando la relación entre estos dos factores.
- **`preprocess_and_split(df, target)`**: Orquesta las transformaciones de datos, excluye la columna de identificación del cliente y prepara los datos para la entrada del modelo, definiendo las columnas numéricas y categóricas para el `ColumnTransformer`.

### 📁 `src/models/model_factory.py`

Este módulo centraliza la creación y configuración de los modelos de machine learning.

- **`create_model_and_params(model_name)`**: Genera una instancia de un modelo de clasificación (ej. **XGBoost, LightGBM, RandomForest**) y su respectiva grilla de hiperparámetros para la optimización. Se han ajustado los rangos de búsqueda de parámetros para explorar combinaciones que maximicen el rendimiento.

### 📁 `src/pipelines/training_pipeline.py`

El corazón del proyecto, donde se ejecuta el flujo de trabajo completo:

- **Detección de Desbalance de Clases**: El pipeline detecta automáticamente si el dataset está desbalanceado.
  - Si el desbalance es significativo, la métrica principal de optimización cambia a **`f1-score`** en lugar de `accuracy`. Esto es crucial, ya que el F1-score es más robusto para evaluar el rendimiento en la clase minoritaria (la de abandono), evitando que un modelo trivial con alta precisión pero baja exhaustividad sea seleccionado.
- **Estrategias de Balanceo de Datos**: Para datasets desbalanceados, se prueban diferentes técnicas de muestreo como **SMOTE**, **ADASYN**, y **Random Oversampling** en conjunto con la optimización de hiperparámetros.
- **`GridSearchCV`**: Se utiliza para realizar una búsqueda exhaustiva de los mejores hiperparámetros para cada modelo, asegurando el mejor rendimiento posible.

### 📁 `src/evaluation/metrics_evaluator.py`

- **`evaluate_model(y_true, y_pred, model_name)`**: Genera el **reporte de clasificación** y la **matriz de confusión** para el mejor modelo, proporcionando una evaluación detallada de su rendimiento en el conjunto de prueba.

## Estrategias Clave para Lograr Mejores Resultados

1.  **Enfoque en F1-Score**: Se prioriza el F1-Score sobre la precisión, lo que garantiza que el modelo sea capaz de identificar eficazmente a los clientes en riesgo de abandono (alta exhaustividad), sin generar un número excesivo de falsos positivos (buena precisión).
2.  **Ingeniería de Características**: La creación de características como el conteo de servicios y la interacción de la permanencia y el contrato mejora la capacidad del modelo para capturar relaciones complejas en los datos.
3.  **Ajuste de Hiperparámetros**: El uso de `GridSearchCV` con rangos de búsqueda ampliados permite encontrar configuraciones óptimas para cada modelo.
4.  **Enfoque Holístico**: El pipeline evalúa múltiples modelos y técnicas de balanceo de clases de manera sistemática, seleccionando el mejor desempeño global para la predicción final.


### Opción 1: ¡Ejecuta el Pipeline en Google Colab!

Puedes ejecutar el pipeline completo, desde la carga de datos hasta la evaluación del mejor modelo, directamente en Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ravilesl/telco-churn-prediction/blob/main/notebooks/colab_runner.ipynb)

Este notebook contiene todo el código necesario, incluyendo la instalación de dependencias, lo que te permite explorar y ejecutar el proyecto sin necesidad de configuraciones locales.

### Opción 2: Ejecutar Localmente
1.  Clona el repositorio: `git clone https://github.com/tu-usuario/telco-churn-prediction.git`
2.  Navega a la carpeta del proyecto: `cd telco-churn-prediction`
3.  Instala las dependencias: `pip install -r requirements.txt`
4.  Ejecuta el pipeline principal: `python src/main.py`
5.  Para iniciar la API, ejecuta:  `uvicorn src.api.main_api:app --host 0.0.0.0 --port 8000`
6.  Entra a este enlace para confirmar que funciona la API: `http://127.0.0.1:8000/`
7.  Entra a este enlace para hacer predicciones: `http://127.0.0.1:8000/docs`
8.  

### Opción 3: Contenerización con Docker (Recomendado)
La forma más robusta y consistente de ejecutar la API es usando Docker. Este método asegura que la aplicación se ejecute en un entorno aislado con todas sus dependencias preinstaladas, evitando conflictos de versión.

#### Requisitos Previos
**Docker Desktop**: Asegúrate de que Docker Desktop está instalado y en ejecución en tu sistema.

#### Archivos Clave
`Dockerfile`: Contiene las instrucciones para construir la imagen de Docker.

`run-docker.ps1`: Un script de PowerShell que automatiza el proceso de construcción y ejecución del contenedor.

#### Pasos para Contenerizar y Ejecutar
**Abre PowerShell**: Navega a la raíz del proyecto donde se encuentran los archivos `Dockerfile` y `run-docker.ps1`.

**Verifica la Configuración**: Asegúrate de que **Docker Desktop** está en ejecución en tu sistema.

**Ejecuta el Script**: Corre el script de PowerShell para construir y lanzar el contenedor con un solo comando. Si es la primera vez que ejecutas un script local en PowerShell, es posible que necesites cambiar la política de ejecución. Puedes hacerlo temporalmente con el siguiente comando (confirma con "S" o "A" si se te pide):

#### PowerShell

`Set-ExecutionPolicy RemoteSigned -Scope Process`
Una vez que la política de ejecución esté configurada, corre el script de esta manera:

`.\run-docker.ps1`
**Verifica la Ejecución**: El script construirá la imagen y lanzará el contenedor. Verás mensajes en la terminal indicando el progreso. Si la ejecución es exitosa, el script te proporcionará la URL para acceder a la API.

#### Acceso a la API Contenerizada
Una vez que el contenedor esté en ejecución, accede a la documentación de la API en tu navegador:
http://localhost:8000/docs