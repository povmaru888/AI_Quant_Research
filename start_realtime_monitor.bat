@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe
  exit /b 1
)
".venv\Scripts\python.exe" tools\start_realtime_monitor.py --host 100.114.147.83 --port 8601
