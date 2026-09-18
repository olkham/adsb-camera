@echo off
rem Start ADS-B Camera. Extra arguments are passed through, e.g.
rem   run.bat --source http://192.168.1.50:8080/stream.mjpeg --port 8091
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No .venv found - run install.bat first.
    exit /b 1
)
if not exist "config.yaml" copy /y "config.example.yaml" "config.yaml" >nul
".venv\Scripts\python.exe" -m adsb_camera -c config.yaml %*
endlocal
