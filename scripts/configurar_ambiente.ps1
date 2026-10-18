$ErrorActionPreference = "Stop"

$raizProjeto = Split-Path -Parent $PSScriptRoot
$ambienteVirtual = Join-Path $raizProjeto ".venv"
$pythonAmbiente = Join-Path $ambienteVirtual "Scripts\python.exe"

Set-Location $raizProjeto

Write-Host "Projeto: $raizProjeto" -ForegroundColor Cyan

if (Get-Command py -ErrorAction SilentlyContinue) {
    $iniciadorPython = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $iniciadorPython = "python"
} else {
    Write-Host "Python não foi encontrado." -ForegroundColor Red
    Write-Host "Instale o Python 3.11 ou 3.12 e marque a opção 'Add Python to PATH'."
    exit 1
}

if (-not (Test-Path $ambienteVirtual)) {
    Write-Host "Criando o ambiente virtual .venv..." -ForegroundColor Yellow

    if ($iniciadorPython -eq "py") {
        & py -3 -m venv $ambienteVirtual
    } else {
        & python -m venv $ambienteVirtual
    }
}

if (-not (Test-Path $pythonAmbiente)) {
    Write-Host "Não foi possível criar o ambiente virtual." -ForegroundColor Red
    exit 1
}

Write-Host "Validando a versão do Python..." -ForegroundColor Yellow
& $pythonAmbiente -c "import sys; print('Python:', sys.version); assert sys.version_info >= (3, 10), 'Utilize Python 3.10 ou superior.'"

Write-Host "Atualizando o instalador de pacotes..." -ForegroundColor Yellow
& $pythonAmbiente -m pip install --upgrade pip setuptools wheel

Write-Host "Instalando as bibliotecas do projeto..." -ForegroundColor Yellow
& $pythonAmbiente -m pip install -r requirements.txt

Write-Host "Registrando o kernel do Jupyter..." -ForegroundColor Yellow
& $pythonAmbiente -m ipykernel install --user `
    --name tcc-visao-computacional `
    --display-name "Python (TCC Visão Computacional)"

Write-Host ""
Write-Host "Ambiente configurado com sucesso." -ForegroundColor Green
Write-Host "Agora execute o arquivo 2_ABRIR_JUPYTER.bat."

