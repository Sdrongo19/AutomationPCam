@echo off
REM Avvio su Windows: crea l'ambiente al primo avvio, poi lancia il programma.
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || exit /b 1
  .venv\Scripts\pip install -r requirements.txt || exit /b 1
)
.venv\Scripts\python main.py %*
