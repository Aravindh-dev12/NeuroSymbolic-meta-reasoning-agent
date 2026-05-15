#!/bin/bash

# NeuroSymbolic AGI Agent - Setup Script
# This script helps set up the environment for the agent

set -e

echo "=========================================="
echo "NeuroSymbolic AGI Agent - Setup"
echo "=========================================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="windows"
    echo "Windows detected. Please use setup.ps1 instead."
    exit 1
else
    echo "Unknown OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [ "$OS" == "linux" ] || [ "$OS" == "macos" ]; then
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for CUDA
echo ""
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    echo "Installing llama-cpp-python with CUDA support..."
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
else
    echo "No NVIDIA GPU detected. Installing CPU version..."
    pip install llama-cpp-python
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs data models cache

# Copy environment example
echo ""
echo "Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
    echo "Please edit .env to configure your settings"
else
    echo ".env file already exists"
fi

# Set permissions
echo ""
echo "Setting permissions..."
chmod +x setup.sh

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the agent:"
echo "  python main.py --interactive"
echo ""
echo "To configure the agent, edit .env file"
echo ""
