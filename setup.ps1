# NeuroSymbolic AGI Agent - Windows Setup Script
# This script helps set up the environment for the agent on Windows

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "NeuroSymbolic AGI Agent - Windows Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check for CUDA
Write-Host ""
Write-Host "Checking for CUDA..." -ForegroundColor Yellow
try {
    nvidia-smi | Out-Null
    Write-Host "NVIDIA GPU detected!" -ForegroundColor Green
    Write-Host "Installing llama-cpp-python with CUDA support..." -ForegroundColor Yellow
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
} catch {
    Write-Host "No NVIDIA GPU detected. Installing CPU version..." -ForegroundColor Yellow
    pip install llama-cpp-python
}

# Create necessary directories
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path data | Out-Null
New-Item -ItemType Directory -Force -Path models | Out-Null
New-Item -ItemType Directory -Force -Path cache | Out-Null

# Copy environment example
Write-Host ""
Write-Host "Setting up environment configuration..." -ForegroundColor Yellow
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env file from .env.example" -ForegroundColor Green
    Write-Host "Please edit .env to configure your settings" -ForegroundColor Yellow
} else {
    Write-Host ".env file already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the virtual environment:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run the agent:" -ForegroundColor Yellow
Write-Host "  python main.py --interactive" -ForegroundColor White
Write-Host ""
Write-Host "To configure the agent, edit .env file" -ForegroundColor Yellow
Write-Host ""
