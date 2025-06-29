@echo off
echo Installing AI Data Analyzer...

:: Check Python version
python --version

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install compatible packages one by one
echo Installing packages...
pip install streamlit
pip install pandas==1.5.3
pip install openpyxl
pip install python-dotenv
pip install openai==0.27.8
pip install langchain==0.0.200
pip install pandasai==1.2.4

echo.
echo Installation complete!
echo.
echo Next steps:
echo 1. Create a .env file with your OpenAI API key
echo 2. Run: streamlit run app.py
pause