@echo off
setlocal

call "%~dp0.venv\Scripts\activate.bat"

streamlit run "%~dp0streamlit_app.py"

endlocal
