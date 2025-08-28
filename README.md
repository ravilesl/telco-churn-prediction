# Predicción de Abandono de Clientes (Telco Churn)

Este proyecto desarrolla un pipeline de Machine Learning de extremo a extremo para predecir el abandono de clientes en una compañía de telecomunicaciones. El pipeline está diseñado con una arquitectura modular para asegurar la reproducibilidad, escalabilidad y mantenibilidad.

## Arquitectura del Proyecto
El proyecto está estructurado en directorios modulares que reflejan las diferentes etapas del ciclo de vida de un proyecto de ciencia de datos:
* `src/utils/`: Contiene módulos para la carga de datos.
* `src/features/`: Módulos para la ingeniería de características y el preprocesamiento de datos.
* `src/models/`: Encapsula la lógica de los modelos y utiliza un patrón Factory para la creación de modelos.
* `src/evaluation/`: Módulo para calcular métricas de rendimiento y generar reportes.
* `src/pipelines/`: Orquesta las etapas de preprocesamiento, modelado y evaluación en un flujo de trabajo cohesivo.
* `src/api/`: Contiene el código de la API para operacionalizar el modelo.
* `tests/`: Módulos para pruebas unitarias.

## Cómo Ejecutar el Proyecto
Este proyecto es totalmente reproducible. Puedes ejecutarlo en tu entorno local o directamente en Google Colab.

### Opción 1: Ejecutar Localmente
1.  Clona el repositorio: `git clone https://github.com/tu-usuario/telco-churn-prediction.git`
2.  Navega a la carpeta del proyecto: `cd telco-churn-prediction`
3.  Instala las dependencias: `pip install -r requirements.txt`
4.  Ejecuta el pipeline principal: `python src/main.py`
5.  Para iniciar la API, ejecuta:  `uvicorn src.api.main_api:app --host 0.0.0.0 --port 8000`

### Opción 2: Ejecutar en Google Colab
Haz clic en el siguiente enlace para abrir el notebook y ejecutar el pipeline completo en la nube.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ravilesl/telco-churn-prediction/blob/main/notebooks/colab_runner.ipynb)
