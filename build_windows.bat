@echo off
REM Crea dist\GestureControl.exe
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || exit /b 1
  .venv\Scripts\pip install -r requirements.txt || exit /b 1
)
.venv\Scripts\pip install -q pyinstaller
.venv\Scripts\pyinstaller --noconfirm gesture_control.spec
