import sys
import os

venv_path = os.environ.get("VIRTUAL_ENV", None)
if venv_path:
    print(f"✅ Entorno virtual activo: {venv_path}")
else:
    print("❌ No hay entorno virtual activo.")
    print(f"Python actual: {sys.executable}")
    