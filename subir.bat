@echo off
setlocal

rem Obtener la fecha y hora actual en un formato de versión
for /f "tokens=2 delims= " %%a in ('date /t') do set "date=%%a"
for /f "tokens=1" %%a in ('time /t') do set "time=%%a"

rem Formatear la fecha y hora para la versión (ej. 2025.08.28-20.05)
set "version=%date:~-4%.%date:~3,2%.%date:~0,2%-%time:^:=.%"
set "version=%version:.=%"

rem Mensaje del commit con la versión dinámica
set "commit_msg=Telco Churm Prediction v3.%version%"

echo Agregando todos los archivos al staging...
git add .

echo Creando un nuevo commit con el mensaje: %commit_msg%
git commit -m "%commit_msg%"

echo Subiendo los cambios a la rama principal...
git push origin main

endlocal