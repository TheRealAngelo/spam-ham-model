@echo off
echo 🎯 SMS Spam Detector - Windows Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found!
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not available
    pause
    exit /b 1
)

echo ✅ pip found!
echo.

REM Install dependencies
echo 🔄 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed!
echo.

REM Train the model
echo 🧠 Training the model...
python train.py
if errorlevel 1 (
    echo ❌ Model training failed
    pause
    exit /b 1
)

echo ✅ Model trained successfully!
echo.

REM Launch the application
echo 🚀 Launching SMS Spam Detector...
echo The application will open in your web browser.
echo Press Ctrl+C to stop the application.
echo.
streamlit run main.py

pause
