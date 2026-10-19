@echo off
echo =====================================================================
echo  Running PMAD Automated Test Suite with Pytest
echo =====================================================================

if exist "C:\Users\BHAVESH BARGAT\AppData\Local\Python\pythoncore-3.11-64\python.exe" (
    "C:\Users\BHAVESH BARGAT\AppData\Local\Python\pythoncore-3.11-64\python.exe" -m pytest tests/ -v
) else (
    py -3.11 -m pytest tests/ -v || python -m pytest tests/ -v
)
pause
