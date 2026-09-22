@echo off
REM Double-click to launch the dashboard. Keep this window open while using it.
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501
pause
