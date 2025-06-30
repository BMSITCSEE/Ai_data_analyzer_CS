# AI-Powered Data Analyzer 📊

An intelligent web application that allows users to analyze CSV and Excel files using natural language queries powered by OpenAI and PandasAI.

## Features

-  **Multi-file Upload**: Support for CSV and Excel files (.csv, .xls, .xlsx)
-  **Sheet Selection**: Navigate through multiple sheets in Excel files
-  **Data Preview**: View top N rows of your data
-  **Natural Language Queries**: Ask questions about your data in plain English
-  **AI-Powered Analysis**: Get intelligent insights using OpenAI GPT models
-  **Query History**: Track all your questions and answers
-  **Feedback System**: Rate AI responses for continuous improvement

##  Installation & Local Run

```bash
# Clone the repository
git clone https://github.com/BMSITCSEE/Ai_data_analyzer_CS.git
cd Ai_data_analyzer_CS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # (Use venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt

# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Run the app
streamlit run app.py

# Live Demo
[Click here to open the deployed app]
(https://ai-data-analyzer-cs.streamlit.app)
