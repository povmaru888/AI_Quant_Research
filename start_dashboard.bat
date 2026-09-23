@echo off
REM Double-click to launch the dashboard. Keep this window open while using it.
cd /d "%~dp0"
set PYEXE=C:\Users\ittraining\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PYEXE%" set PYEXE=py
"%PYEXE%" -m streamlit run app.py --server.port 8501
pause
