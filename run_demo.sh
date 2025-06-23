#!/bin/bash

echo "Setting up environment for QuantJourney Technical Indicators..."

# Step 0: Check for uv and install if not available
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing with pipx..."
    if ! command -v pipx &> /dev/null; then
        echo "pipx not found. Installing pipx..."
        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
        export PATH="$HOME/.local/bin:$PATH"
    fi
    pipx install uv
else
    echo "uv is already installed: $(uv --version)"
fi

# Step 1: Create virtual environment with Python 3.11
echo "Creating virtual environment..."
uv venv --python 3.11

# Step 2: Activate virtual environment
source .venv/bin/activate
echo "Activated virtualenv with Python: $(python --version)"

# Step 3: Install project with dev dependencies
echo "Installing project with dev dependencies..."
uv pip install -e ".[dev]"

# Step 4: Run the example script
echo "Running the indicators demo..."
python examples/run_indicators.py
