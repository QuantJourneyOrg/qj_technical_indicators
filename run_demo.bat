@echo off
echo Setting up environment for QuantJourney Technical Indicators...

:: Step 0: Check if uv is installed
where uv >nul 2>nul
if errorlevel 1 (
    echo uv is not installed.
    echo Installing uv using pipx...

    where pipx >nul 2>nul
    if errorlevel 1 (
        echo pipx not found. Installing pipx first...
        python -m pip install --user pipx
        python -m pipx ensurepath
        set PATH=%USERPROFILE%\.local\bin;%PATH%
    )

    pipx install uv
) else (
    echo uv is already installed.
)

:: Step 1: Create virtual environment
echo Creating virtual environment...
uv venv --python 3.11

:: Step 2: Activate virtual environment
call .venv\Scripts\activate.bat
echo Activated virtualenv with Python:
python --version

:: Step 3: Install project with dev dependencies
echo Installing project with dev dependencies...
uv pip install -e ".[dev]"

:: Step 4: Run the example script
echo Running the indicators demo...
python examples\run_indicators.py
