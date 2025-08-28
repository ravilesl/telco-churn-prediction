# ==============================================================================
# Script de Automatización para Docker (PowerShell)
# Construye y ejecuta una imagen de Docker para la API de FastAPI.
# Incluye validaciones y gestión de errores específicas para PowerShell.
# ==============================================================================

# 1. Configuración de Variables
# ------------------------------------------------------------------------------
# Genera una etiqueta de imagen única usando la fecha y hora
$imageName = "telco-api-$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss')"
$containerName = "telco-api-container"
$portMapping = "8000:8000"
$requirementsFile = "requirements.txt"
$dockerfile = "Dockerfile"

Write-Host "⚙️  Variables de configuración:" -ForegroundColor Green
Write-Host "   - Nombre de la imagen: $imageName"
Write-Host "   - Nombre del contenedor: $containerName"
Write-Host "   - Mapeo de puertos: $portMapping"
Write-Host "--------------------------------------------------------"

# 2. Validaciones de Seguridad y Pre-ejecución
# ------------------------------------------------------------------------------
# Comprueba si Docker está en ejecución
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Error: Docker no parece estar en ejecución. Por favor, inicia el servicio de Docker e inténtalo de nuevo." -ForegroundColor Red
    exit
}

# Comprueba si los archivos críticos existen
if (-not (Test-Path $dockerfile)) {
    Write-Host "❌ Error: No se encontró el archivo '$dockerfile' en el directorio actual. Asegúrate de que existe." -ForegroundColor Red
    exit
}

if (-not (Test-Path $requirementsFile)) {
    Write-Host "❌ Error: No se encontró el archivo '$requirementsFile' en el directorio actual." -ForegroundColor Red
    exit
}

# Detiene y elimina cualquier contenedor existente con el mismo nombre
$existingContainer = docker ps -a --format "{{.Names}}" --filter "name=$containerName"
if ($existingContainer) {
    Write-Host "⚠️  Contenedor existente encontrado. Deteniendo y eliminando '$containerName'..." -ForegroundColor Yellow
    docker stop $containerName | Out-Null
    docker rm $containerName | Out-Null
}

# 3. Construcción de la Imagen
# ------------------------------------------------------------------------------
Write-Host "🏗️  Construyendo la imagen de Docker..." -ForegroundColor Green
docker build -t $imageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: La construcción de la imagen ha fallado. Revisar los logs de arriba." -ForegroundColor Red
    exit
}

Write-Host "✅ Imagen construida exitosamente: $imageName" -ForegroundColor Green
Write-Host "--------------------------------------------------------"

# 4. Ejecución del Contenedor
# ------------------------------------------------------------------------------
Write-Host "🚀 Ejecutando la API en el puerto $portMapping..." -ForegroundColor Green
docker run -d --name $containerName -p $portMapping $imageName
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: No se pudo iniciar el contenedor." -ForegroundColor Red
    exit
}

Write-Host "✅ Contenedor '$containerName' iniciado. La API está disponible en http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Para ver los logs del contenedor, ejecuta: docker logs -f $containerName"
Write-Host "Para detener el contenedor, ejecuta: docker stop $containerName"
