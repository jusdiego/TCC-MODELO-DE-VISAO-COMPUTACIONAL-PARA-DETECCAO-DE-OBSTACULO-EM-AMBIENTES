@echo off
cd /d "%~dp0"

echo Configurando o ambiente Python do TCC...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\configurar_ambiente.ps1"

if errorlevel 1 (
    echo.
    echo A configuracao encontrou um erro. Leia a mensagem acima.
)

echo.
pause

