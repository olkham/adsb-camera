@echo off
rem Set up ADS-B Camera: pick a Python, create .venv, install requirements,
rem create config.yaml and the data folder.
rem   install.bat        interactive
rem   install.bat -y     recreate .venv without asking
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "FORCE=0"
if /i "%~1"=="-y" set "FORCE=1"
if /i "%~1"=="--force" set "FORCE=1"

set "CHECK=import sys; print('.'.join(map(str, sys.version_info[:3]))) if sys.version_info[:2] >= (3, 9) else None"

rem ---- find interpreters (the Microsoft Store "python3" stub prints nothing, so it is ignored)
set "V_PY="
set "V_PY3="
where python >nul 2>&1 && for /f "delims=" %%v in ('python -c "%CHECK%" 2^>nul') do set "V_PY=%%v"
where python3 >nul 2>&1 && for /f "delims=" %%v in ('python3 -c "%CHECK%" 2^>nul') do set "V_PY3=%%v"

set "PY="
if defined V_PY if defined V_PY3 if not "!V_PY!"=="!V_PY3!" (
    echo Two different Python versions were found:
    echo   1^) python   !V_PY!
    echo   2^) python3  !V_PY3!
    set "CHOICE=1"
    set /p "CHOICE=Which one should create the .venv? [1/2] (default 1): "
    if "!CHOICE!"=="1" set "PY=python"
    if "!CHOICE!"=="2" set "PY=python3"
    if not defined PY (
        echo Please answer 1 or 2.
        exit /b 2
    )
)
if not defined PY if defined V_PY set "PY=python"
if not defined PY if defined V_PY3 set "PY=python3"
if not defined PY (
    rem last resort: the Windows py launcher
    for /f "delims=" %%v in ('py -3 -c "%CHECK%" 2^>nul') do set "V_LAUNCHER=%%v"
    if defined V_LAUNCHER set "PY=py -3"
)
if not defined PY (
    echo [x] No Python 3.9+ found as python or python3.
    echo     Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    exit /b 1
)
for /f "delims=" %%v in ('%PY% -c "%CHECK%"') do set "PYVER=%%v"
echo ==^> Using %PY% %PYVER%

rem ---- virtual environment
if exist ".venv" (
    set "REPLY=n"
    if "%FORCE%"=="1" (set "REPLY=y") else (set /p "REPLY=.venv already exists. Recreate it? [y/N]: ")
    if /i "!REPLY!"=="y" (
        echo ==^> Removing old .venv
        rmdir /s /q ".venv"
    ) else (
        echo ==^> Keeping existing .venv ^(dependencies will be updated^)
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo ==^> Creating .venv
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [x] Could not create the virtual environment.
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

echo ==^> Upgrading pip
"%VPY%" -m pip install --upgrade pip wheel >nul
echo ==^> Installing requirements ^(numpy, scipy, opencv, fastapi, paho-mqtt ...^)
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [x] pip install failed.
    exit /b 1
)

rem ---- project files
if not exist "data" mkdir "data"
if not exist "config.yaml" (
    copy /y "config.example.yaml" "config.yaml" >nul
    echo ==^> Created config.yaml - set video.source, mqtt.host and site.lat/lon
) else (
    echo ==^> config.yaml already exists - left unchanged
)
if not exist ".env" copy /y ".env.example" ".env" >nul

echo ==^> Checking the install
"%VPY%" -c "import cv2, numpy, scipy, fastapi, uvicorn, paho.mqtt, yaml; print('  OpenCV', cv2.__version__, '| numpy', numpy.__version__)"
if errorlevel 1 exit /b 1

echo.
echo ==^> Done. Next:
echo     1. edit config.yaml ^(camera stream, MQTT broker, camera location^)
echo     2. run.bat
echo     3. open http://localhost:8090
endlocal
