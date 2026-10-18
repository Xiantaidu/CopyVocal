@echo off
setlocal
cd /d "%~dp0"
if not exist "python\pythonw.exe" (
  echo Missing local Python runtime. Run scripts\bootstrap.py first.
  exit /b 1
)
start "" "python\pythonw.exe" -X utf8 app_gui.py %*
exit /b 0
