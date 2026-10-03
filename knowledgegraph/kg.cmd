@echo off
rem Windows launcher equivalent of ./kg. Project JSON and logs are UTF-8 on every platform.
setlocal
set "PYTHONUTF8=1"
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\cli.py" %*
exit /b %ERRORLEVEL%
