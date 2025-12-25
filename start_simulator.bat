@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo    CMSE443 Quadcopter Simulator - Automatic Setup & Launch
echo ============================================================
echo.

:: Get the directory where the batch file is located
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [1/6] Checking Python installation...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Display Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo [OK] Found: %PYTHON_VER%

echo.
echo [2/6] Checking/Creating virtual environment...
if not exist "%PROJECT_DIR%venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

echo.
echo [3/6] Activating virtual environment...
call "%PROJECT_DIR%venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.

echo.
echo [4/6] Installing/Updating dependencies...

:: Upgrade pip first
python -m pip install --upgrade pip --quiet

:: IMPORTANT: Remove conflicting package rpc-msgpack (requires tornado>=6.1)
:: We need msgpack-rpc-python which requires tornado<5
echo Removing conflicting packages...
pip uninstall rpc-msgpack -y --quiet 2>nul

:: Install tornado 4.5.3 first (required by msgpack-rpc-python)
echo Installing tornado (compatible version 4.5.3)...
pip install tornado==4.5.3 --force-reinstall --quiet

:: Install msgpack-rpc-python (requires tornado<5)
echo Installing msgpack-rpc-python...
pip install msgpack-rpc-python --quiet

echo Installing numpy...
pip install numpy --quiet
if %errorlevel% neq 0 (
    echo [WARNING] numpy installation had issues, continuing...
)

echo Installing cosysairsim...
pip install cosysairsim --quiet
if %errorlevel% neq 0 (
    echo [WARNING] cosysairsim installation had issues, continuing...
)

echo Installing XInput-Python (for Xbox controller)...
pip install XInput-Python --quiet
if %errorlevel% neq 0 (
    echo [WARNING] XInput-Python installation had issues (optional for keyboard fallback)
)

echo Installing keyboard library...
pip install keyboard --quiet
if %errorlevel% neq 0 (
    echo [WARNING] keyboard installation had issues
)

echo Installing nest_asyncio...
pip install nest_asyncio --quiet

:: Install from requirements.txt if it exists (but ignore tornado conflicts)
if exist "%PROJECT_DIR%requirements.txt" (
    echo Installing from requirements.txt...
    pip install -r "%PROJECT_DIR%requirements.txt" --quiet 2>nul
)

echo [OK] Dependencies installed.

echo.
echo [5/6] Starting AirSim Simulator (Blocks environment)...

:: Check if Blocks.exe exists
if not exist "%PROJECT_DIR%Windows\Blocks.exe" (
    echo [ERROR] Blocks.exe not found at: %PROJECT_DIR%Windows\Blocks.exe
    echo Please ensure the AirSim Blocks environment is in the Windows folder.
    pause
    exit /b 1
)

:: Start Blocks.exe in the background
start "" "%PROJECT_DIR%Windows\Blocks.exe"
echo [OK] AirSim Blocks started. Waiting for it to initialize...

:: Wait for AirSim to start (giving it time to load)
echo Waiting 15 seconds for AirSim to initialize...
timeout /t 15 /nobreak >nul

echo.
echo [6/6] Starting Quadcopter Controller...
echo.
echo ============================================================
echo    Controller is starting - Use GUI to control the drone
echo ============================================================
echo.
echo Controls:
echo   Xbox Controller:
echo     - Left Stick: Pitch/Roll
echo     - Right Stick: Yaw  
echo     - RT: Ascend, LT: Descend
echo     - A: Arm/Disarm, B: Emergency Stop
echo.
echo   Keyboard Fallback:
echo     - WASD: Pitch/Roll
echo     - Q/E: Yaw left/right
echo     - Space/Shift: Ascend/Descend
echo     - Enter: Arm/Disarm, Esc: Emergency Stop
echo.
echo ============================================================
echo.

:: Run the main Python application
python "%PROJECT_DIR%main.py"

:: Deactivate virtual environment when done
call deactivate

echo.
echo [DONE] Simulator closed.
pause
