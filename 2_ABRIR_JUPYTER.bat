@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo O ambiente ainda nao foi configurado.
    echo Execute primeiro o arquivo 1_CONFIGURAR_AMBIENTE.bat.
    echo.
    pause
    exit /b 1
)

echo Abrindo o Jupyter Lab...
".venv\Scripts\python.exe" -m jupyter lab

echo.
pause

