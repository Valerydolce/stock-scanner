@echo off
REM ============================================================
REM run_scanner.bat — Windows Task Scheduler entry point
REM Place this in the same folder as stock_scanner.py
REM ============================================================

REM Move to the folder this .bat lives in (so relative paths work)
cd /d "%~dp0"

REM Prefer the project venv's python; fall back to system python
set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "LOG=%~dp0stock_scanner.log"

echo. >> "%LOG%"
echo ============================== >> "%LOG%"
echo Run at %date% %time% >> "%LOG%"
echo ============================== >> "%LOG%"

"%PYTHON%" "%~dp0stock_scanner.py" >> "%LOG%" 2>&1

REM Exit code 0 = success (Task Scheduler shows "completed successfully")
REM Anything else = failure (Task Scheduler shows last result as nonzero)
exit /b %ERRORLEVEL%