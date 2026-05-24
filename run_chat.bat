@echo off
REM Start Kokomi Chat - Automatically activates venv
cd /d "%~dp0"

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║     KOKOMI - Local AI Assistant                      ║
echo ║     Starting chat interface...                        ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM Activate venv and run chat
call venv\Scripts\activate.bat
python main.py

pause
