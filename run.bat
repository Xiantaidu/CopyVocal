@echo off
setlocal
cd /d "%~dp0"
if not exist "python\python.exe" (
  echo Missing local Python runtime. Run scripts\bootstrap.py first.
  exit /b 1
)
"python\python.exe" -X utf8 -u "scripts\copyvocal_cli.py" %*
exit /b %errorlevel%
