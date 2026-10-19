@echo off
echo =====================================================================
echo  Predictive Maintenance & Anomaly Detection (PMAD) Dashboard
echo =====================================================================
echo Starting Streamlit Web Application...

if exist "C:\Users\BHAVESH BARGAT\AppData\Local\Python\pythoncore-3.11-64\python.exe" (
    "C:\Users\BHAVESH BARGAT\AppData\Local\Python\pythoncore-3.11-64\python.exe" -m streamlit run app.py
) else (
    py -3.11 -m streamlit run app.py || python -m streamlit run app.py
)
pause
