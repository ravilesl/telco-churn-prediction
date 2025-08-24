# Usa una imagen base de Python
FROM python:3.13.5

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requerimientos e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente del proyecto
COPY . .

# Expone el puerto que usará FastAPI
EXPOSE 8000

# Crea un usuario no-root para correr la aplicación, mejorando la seguridad
RUN adduser --system --no-create-home appuser
USER appuser

# Comando para correr la aplicación con Uvicorn
CMD ["uvicorn", "src.api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]