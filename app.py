import streamlit as st
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
import openai
import tempfile
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re

# Loading environment variables
load_dotenv()

# Adding callbacks
def save_positive_feedback(item_id):
    st.session_state.feedback[item_id] = 'positive'
    
def save_negative_feedback(item_id):
    st.session_state.feedback[item_id] = 'negative'


def generate_graph(df, prompt, ai_suggestion=None):
    """Generate graphs based on user prompt"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Style
    sns.set_style("whitegrid")
    
    prompt_lower = prompt.lower()

    # selecting columns
    def get_best_columns(df, column_type='numeric'):
        id_patterns = ['id', 'index', 'key', 'code', '_id']
        
        if column_type == 'numeric':
            cols = df.select_dtypes(include=['float64', 'int64']).columns
            # Filtering id columns
            cols = [c for c in cols if not any(pattern in c.lower() for pattern in id_patterns)]
            return cols if cols else df.select_dtypes(include=['float64', 'int64']).columns
        else:
            cols = df.columns.tolist()
            # Prioritizing non-ID columns
            cols = [c for c in cols if not any(pattern in c.lower() for pattern in id_patterns)]
            return cols if cols else df.columns.tolist()
    
    # Detecting type of graph created
    if any(word in prompt_lower for word in ['histogram', 'distribution', 'spread']):
        # Finding numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]  
            df[col].hist(ax=ax, bins=30, edgecolor='black')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
    elif any(word in prompt_lower for word in ['bar chart', 'bar graph', 'bar plot']):
        # Creating a bar chart
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            # Grouping by first column and taking mean of second
            grouped = df.groupby(x_col)[y_col].mean().head(20)
            grouped.plot(kind='bar', ax=ax)
            ax.set_title(f'Average {y_col} by {x_col}')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45)
            
    elif any(word in prompt_lower for word in ['scatter', 'correlation']):
        # Creating scatter plot
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            ax.scatter(df[x_col], df[y_col], alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'{x_col} vs {y_col}')
            
    elif any(word in prompt_lower for word in ['pie chart', 'pie graph']):
        # Creating pie chart
        if len(df.columns) >= 1:
            col = df.columns[0]
            value_counts = df[col].value_counts().head(10)
            value_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title(f'Distribution of {col}')
            ax.set_ylabel('')
            
    elif any(word in prompt_lower for word in ['line chart', 'line graph', 'trend']):
        # Creating line chart
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]: 
                df[col].plot(ax=ax, label=col)
            ax.set_title('Trend Analysis')
            ax.set_xlabel('Index')
            ax.legend()
            
    elif any(word in prompt_lower for word in ['heatmap', 'correlation matrix']):
        # Creating correlation heatmap
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap')
            
    else:
        # Default:Ccreating a bar chart 
        col = df.columns[0]
        df[col].value_counts().head(15).plot(kind='bar', ax=ax)
        ax.set_title(f'Top Values in {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_base64

# Page configuration
st.set_page_config(
    page_title="AI Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .file-upload-section {
        background-color: #F3F4F6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .data-preview-section {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .prompt-section {
        background-color: #F0FDF4;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .answer-box {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .history-item {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stButton > button {
        border-radius: 5px;
        font-weight: 500;
    }
    .uploaded-file-card {
        background-color: #FFFFFF;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border: 1px solid #E5E7EB;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# giving values to session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# Header
st.markdown('<h1 class="main-header">📊 AI-Powered Data Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6B7280; font-size: 1.1rem;">Upload your CSV or Excel files and ask questions in natural language</p>', unsafe_allow_html=True)

# Initializing OpenAI
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("⚠️ OpenAI API key not found! Please add it to your .env file.")
    st.info("To get started: Create a `.env` file in your project root and add: `OPENAI_API_KEY=your_api_key_here`")
    st.stop()

try:
    openai.api_key = api_key
    openai.project = "proj_T1EU0OZyEmys7aLRxmpS7OJG"  

except Exception as e:
    st.error(f"Error initializing OpenAI: {str(e)}")
    st.stop()

# File Upload Section
st.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">📁 Upload Files</h2>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose CSV or Excel files",
    type=['csv', 'xls', 'xlsx'],
    accept_multiple_files=True,
    help="You can upload multiple files at once. Supported formats: CSV, XLS, XLSX"
)

if uploaded_files:
    new_files = 0
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            new_files += 1
            try:
                # Saving file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                    tmp_file.write(file.getvalue())
                    temp_path = tmp_file.name
                
                # Loading file 
                if file.name.endswith('.csv'):
                    df = pd.read_csv(temp_path)
                    st.session_state.uploaded_files[file.name] = {
                        'path': temp_path,
                        'sheets': {'Sheet1': df},
                        'type': 'csv',
                        'size': file.size
                    }
                else:
                    excel_file = pd.ExcelFile(temp_path)
                    sheets = {}
                    for sheet_name in excel_file.sheet_names:
                        sheets[sheet_name] = pd.read_excel(temp_path, sheet_name=sheet_name)
                    st.session_state.uploaded_files[file.name] = {
                        'path': temp_path,
                        'sheets': sheets,
                        'type': 'excel',
                        'size': file.size
                    }
            except Exception as e:
                st.error(f"Error loading {file.name}: {str(e)}")
    
    if new_files > 0:
        st.success(f"✅ Successfully uploaded {new_files} new file(s)")

# Displaying list of uploaded files
if st.session_state.uploaded_files:
    st.markdown("### 📂 Uploaded Files")
    for filename, file_info in st.session_state.uploaded_files.items():
        file_type_icon = "📊" if file_info['type'] == 'excel' else "📄"
        sheets_count = len(file_info['sheets'])
        size_mb = file_info['size'] / (1024 * 1024)
        
        st.markdown(f"""
        <div class="uploaded-file-card">
            <div>
                <strong>{file_type_icon} {filename}</strong>
                <span style="color: #6B7280; font-size: 0.9rem;">
                    ({sheets_count} sheet{'s' if sheets_count > 1 else ''}, {size_mb:.1f} MB)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Displaying uploaded files and data preview
if st.session_state.uploaded_files:
    st.markdown('<div class="data-preview-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">📋 File Selection & Preview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_file = st.selectbox(
            "Select a file",
            options=list(st.session_state.uploaded_files.keys()),
            help="Choose which file to analyze"
        )
    
    with col2:
        if selected_file:
            file_data = st.session_state.uploaded_files[selected_file]
            sheet_names = list(file_data['sheets'].keys())
            selected_sheet = st.selectbox(
                "Select a sheet",
                options=sheet_names,
                help="Choose which sheet to analyze"
            )
    
    with col3:
        n_rows = st.number_input(
            "Top N rows",
            min_value=1,
            max_value=100,
            value=5,
            help="Number of rows to preview"
        )
    
    # Displaying data preview
    if selected_file and selected_sheet:
        st.session_state.current_df = st.session_state.uploaded_files[selected_file]['sheets'][selected_sheet]
        
        st.markdown("### 👀 Data Preview")
	
            # Showing column info
        with st.expander("📊 Column Information", expanded=False):
            col_info = pd.DataFrame({
                'Column': st.session_state.current_df.columns,
                'Type': st.session_state.current_df.dtypes.astype(str),
                'Non-Null Count': st.session_state.current_df.count(),
                'Null Count': st.session_state.current_df.isnull().sum(),
                'Unique Values': st.session_state.current_df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Displaying data preview
        st.dataframe(
            st.session_state.current_df.head(n_rows),
            use_container_width=True,
            height=300
        )
        
        # Displaying basic statistics
        st.markdown("### 📈 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(st.session_state.current_df):,}")
        with col2:
            st.metric("Total Columns", len(st.session_state.current_df.columns))
        with col3:
            memory_mb = st.session_state.current_df.memory_usage().sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")
        with col4:
            null_count = st.session_state.current_df.isnull().sum().sum()
            null_percentage = (null_count / (len(st.session_state.current_df) * len(st.session_state.current_df.columns))) * 100
            st.metric("Missing Values", f"{null_count:,} ({null_percentage:.1f}%)")
        
        # Additional statistics for numeric columns
        numeric_cols = st.session_state.current_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            with st.expander("📊 Numeric Column Statistics", expanded=False):
                st.dataframe(
                    st.session_state.current_df[numeric_cols].describe(),
                    use_container_width=True
                )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prompt Section
    if st.session_state.current_df is not None:
        st.markdown('<div class="prompt-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">💬 Ask Questions About Your Data</h2>', unsafe_allow_html=True)
        
        
        # Example prompts
        example_prompts = [
    	"What are the key statistics for this dataset?",
    	"Show me sales trends over time",
    	"Which categories have the highest values?",
    	"Display customer distribution by region",
    	"Analyze the relationship between price and quantity",
    	"Create a summary report of all numeric columns",
    	"Find patterns in the data",
    	"What insights can you provide about this data?",
    	"Generate a bar chart showing top 10 items",
    	"Create a histogram of the age distribution"
	]
        
        with st.expander("💡 Example Questions", expanded=False):
            for prompt in example_prompts:
                st.markdown(f"• {prompt}")
        
        user_prompt = st.text_area(
            "Enter your question",
            placeholder="e.g., What is the average sales by region? Show me the top 5 customers by revenue.",
            height=100,
            help="Ask any question about your data in natural language"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Clear Question", use_container_width=True):
                st.rerun()
        
        if analyze_button and user_prompt:
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    df = st.session_state.current_df
            
                    # Checking if user wants graph
                    graph_keywords = ['plot', 'graph', 'chart', 'visualize', 'visualization', 'show me', 'display', 
                            'histogram', 'scatter', 'bar', 'pie', 'line', 'heatmap', 'correlation']
                    is_graph_request = any(keyword in user_prompt.lower() for keyword in graph_keywords)
            
                    # Preparing data sample 
                    df_sample = df.head(50).to_csv(index=False) if len(df) > 50 else df.to_csv(index=False)
            
                    # Creating prompt for AI
                    system_prompt = """You are a professional data analyst. Follow these rules strictly:
		    1. For numerical questions, provide exact numbers and calculations
 		    2. For 'top N' questions, show actual values in a clean list format
	            3. For graph requests, describe what the visualization would show
		    4. Never show code, functions, or technical objects like '<lambda>'
	            5. Format answers as brief, professional insights
		    6. If data has computed columns, explain what they represent in plain English"""
            
                    prompt = f"""Here's a dataset with {len(df)} rows and {len(df.columns)} columns.
                    Columns: {list(df.columns)}
                    Data types: {dict(df.dtypes.astype(str))}
            
                    Sample data:
                    {df_sample}
            
                    User question: {user_prompt}"""
            
                    # Getting AI response
                    response = openai.ChatCompletion.create(
                         model="gpt-3.5-turbo",
                         messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                         ],
                         temperature=0.3,
                         max_tokens=800
                    )
                    ai_response = response['choices'][0]['message']['content']
                
                    # Cleaning the response 
                    
                    if any(term in str(ai_response).lower() for term in ['<function', 'lambda', 'object at', 'dtype']):
                        ai_response = "I found some computed values in the data. Let me provide a clearer analysis:\n\n" + \
                                  "The dataset contains processed columns that need proper evaluation. " + \
                                  "Please ensure all calculated fields are properly resolved before analysis."
                
                    # Removing code-like patterns
                    ai_response = re.sub(r'<[^>]+>', '', ai_response)  # Remove HTML-like tags
                    ai_response = re.sub(r'\b0x[0-9a-fA-F]+\b', '', ai_response)  # Remove memory addresses
                
                
                    # Adding to history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    history_item = {
                        'timestamp': timestamp,
                        'file': selected_file,
                        'sheet': selected_sheet,
                        'question': user_prompt,
                        'answer': ai_response,
                        'id': len(st.session_state.prompt_history)
                    }
                    st.session_state.prompt_history.append(history_item)
            
                    # Displaying answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Answer")
                    st.write(ai_response)
            
                    # Generating graph if requested
                    if is_graph_request:
                        try:
                            st.markdown("### 📊 Generated Visualization")
                            fig = generate_graph(df, user_prompt, ai_response)
                            st.pyplot(fig)
                    
                             # Adding download button for the graph
                            img_base64 = fig_to_base64(fig)
                            href = f'<a href="data:image/png;base64,{img_base64}" download="chart.png">Download Chart</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    
                        except Exception as e:
                            st.warning(f"Could not generate automatic visualization: {str(e)}")
                    
                            # Offering manual graph options
                            st.markdown("#### 📊 Manual Visualization Options")
                            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                            col1, col2 = st.columns(2)
                            with col1:
                                chart_type = st.selectbox("Select chart type", 
                                ["Bar Chart", "Histogram", "Scatter Plot", "Line Chart", "Pie Chart", "Heatmap"])
                    
                            with col2:
                                if chart_type in ["Histogram", "Line Chart"]:
                                    selected_col = st.selectbox("Select column", numeric_cols if numeric_cols else df.columns.tolist())
                                elif chart_type == "Scatter Plot":
                                    x_col = st.selectbox("X axis", numeric_cols if numeric_cols else df.columns.tolist())
                                    y_col = st.selectbox("Y axis", numeric_cols if numeric_cols else df.columns.tolist())
                                else:
                                    selected_col = st.selectbox("Select column", df.columns.tolist())
                    
                            if st.button("Generate Chart"):
                                fig, ax = plt.subplots(figsize=(10, 6))
                        
                                if chart_type == "Histogram" and selected_col:
                                    df[selected_col].hist(ax=ax, bins=30, edgecolor='black')
                                    ax.set_title(f'Distribution of {selected_col}')
                            
                                elif chart_type == "Bar Chart" and selected_col:
                                    df[selected_col].value_counts().head(20).plot(kind='bar', ax=ax)
                                    ax.set_title(f'Top Values in {selected_col}')
                                    plt.xticks(rotation=45)
                            
                                elif chart_type == "Scatter Plot" and 'x_col' in locals() and 'y_col' in locals():
                                    ax.scatter(df[x_col], df[y_col], alpha=0.6)
                                    ax.set_xlabel(x_col)
                                    ax.set_ylabel(y_col)
                                    ax.set_title(f'{x_col} vs {y_col}')
                            
                                elif chart_type == "Line Chart" and selected_col:
                                    df[selected_col].plot(ax=ax)
                                    ax.set_title(f'Trend of {selected_col}')
                            
                                elif chart_type == "Pie Chart" and selected_col:
                                    df[selected_col].value_counts().head(10).plot(kind='pie', ax=ax, autopct='%1.1f%%')
                                    ax.set_title(f'Distribution of {selected_col}')
                            
                                elif chart_type == "Heatmap":
                                    numeric_df = df.select_dtypes(include=['float64', 'int64'])
                                    if not numeric_df.empty:
                                        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                                        ax.set_title('Correlation Heatmap')
                        
                                plt.tight_layout()
                                st.pyplot(fig)
            

            st.markdown("---")# Feedback buttons with callbacks
            st.markdown("**Was this answer helpful?**")
            col1, col2, col3 = st.columns([1, 1, 8])
            
            current_item_id = history_item['id']
            
            with col1:
                if st.button("👍 Yes", 
                             key=f"pos_{current_item_id}", 
                             on_click=save_positive_feedback,
                             args=(current_item_id,),
                             help="This answer was helpful"):
                    st.success("Thanks for your feedback!")
                    
            with col2:
                if st.button("👎 No", 
                             key=f"neg_{current_item_id}",
                             on_click=save_negative_feedback,
                             args=(current_item_id,),
                             help="This answer needs improvement"):
                    st.info("Thanks for your feedback! We'll work on improving.")
            
            # Show if already rated
            if current_item_id in st.session_state.feedback:
                with col3:
                    if st.session_state.feedback[current_item_id] == 'positive':
                        st.markdown("✅ _You found this helpful_")
                    else:
                        st.markdown("📝 _You suggested improvement_")
                    
            
            
                except Exception as e:
                    error_msg = str(e)
                    if "api" in error_msg.lower():
			    st.error("⚠️ Analysis service temporarily unavailable. Please try again.")
		    elif "data" in error_msg.lower():
			    st.error("📊 Data format issue detected. Please check your file structure.")
		    else:
			    st.error("❌ Unable to process this request. Try rephrasing your question.")

    elif analyze_button and not user_prompt:
        st.warning("⚠️ Please enter a question to analyze your data.")

    st.markdown('</div>', unsafe_allow_html=True)

# Prompt History 
if st.session_state.prompt_history:
    st.markdown("---")
    with st.expander("📜 Query History", expanded=False):
        st.markdown("### Previous Questions & Answers")
        
        # Adding filter options
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            filter_file = st.selectbox(
                "Filter by file",
                options=["All"] + list(set(item['file'] for item in st.session_state.prompt_history)),
                key="history_filter_file"
            )
        with col2:
            filter_feedback = st.selectbox(
                "Filter by feedback",
                options=["All", "Positive", "Negative", "No feedback"],
                key="history_filter_feedback"
            )
        
        # Filtering history
        filtered_history = st.session_state.prompt_history.copy()
        if filter_file != "All":
            filtered_history = [item for item in filtered_history if item['file'] == filter_file]
        
        if filter_feedback != "All":
            if filter_feedback == "Positive":
                filtered_history = [item for item in filtered_history if st.session_state.feedback.get(item['id']) == 'positive']
            elif filter_feedback == "Negative":
                filtered_history = [item for item in filtered_history if st.session_state.feedback.get(item['id']) == 'negative']
            else:  
                filtered_history = [item for item in filtered_history if item['id'] not in st.session_state.feedback]
        
        
        # Displaying filtered history
	for item in reversed(filtered_history):
	    # Checking feedback status
	    feedback = st.session_state.feedback.get(item['id'], None)
	    
	    # Color code based on feedback
	    border_color = "#10B981" if feedback == 'positive' else "#EF4444" if feedback == 'negative' else "#3B82F6"
	    
	    st.markdown(f'<div class="history-item" style="border-left: 4px solid {border_color};">', unsafe_allow_html=True)
	    
	    col1, col2 = st.columns([5, 1])
	    with col1:
	        st.markdown(f"**🕐 {item['timestamp']}** | 📁 {item['file']} - {item['sheet']}")
	        st.markdown(f"**Q:** {item['question']}")
	        st.markdown(f"**A:** {item['answer']}")
	    
	    with col2:
	        if feedback == 'positive':
	            st.markdown("✅ **Helpful**")
	        elif feedback == 'negative':
	            st.markdown("❌ **Needs Work**")
	        else:
	            st.markdown("⚪ _No rating_")
	    
	    st.markdown('</div>', unsafe_allow_html=True)
	    st.markdown("---")
        
        # Exporting history button
        if st.button("📥 Export History to JSON"):
            history_export = {
                'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_queries': len(st.session_state.prompt_history),
                'queries': st.session_state.prompt_history,
                'feedback': st.session_state.feedback
            }
            st.download_button(
                label="Download History",
                data=json.dumps(history_export, indent=2),
                file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Sidebar with instructions 
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. ** Upload Files**: Upload one or more CSV or Excel files
    2. ** Select Data**: Choose a file and sheet to analyze
    3. ** Preview**: View the top N rows of your data
    4. ** Ask Questions**: Type questions in natural language
    5. ** Get Insights**: AI will analyze and answer your questions
    6. ** Give Feedback**: Rate responses with thumbs up or down
    """)
    
    st.markdown("---")
    
    st.markdown("## 💡 Example Questions")
    st.markdown("""
    **Basic Analysis:**
    - What are the column names and types?
    - Show me basic statistics
    - How many null values are there?
    
    **Aggregations:**
    - What is the sum/average/count by category?
    - Show me the top 10 rows by value
    - Group by column and calculate mean
    
    **Data Exploration:**
    - Find correlations between columns
    - Show unique values in a column
    - Filter rows where condition is met

    **📊 Visualizations:**
    - Create a bar chart of sales by region
    - Show me a histogram of age distribution
    - Plot a scatter plot of price vs quantity
    - Generate a pie chart of product categories
    - Create a line chart showing trends over time
    - Show me a heatmap of correlations

    **Advanced Analysis:**
    - Create a pivot table
    - Calculate percentage change
    - Find outliers in the data
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Current Session Stats")
    if st.session_state.uploaded_files:
        st.metric("Files Uploaded", len(st.session_state.uploaded_files))
        st.metric("Total Queries", len(st.session_state.prompt_history))
        
        if st.session_state.feedback:
            positive = sum(1 for f in st.session_state.feedback.values() if f == 'positive')
            negative = sum(1 for f in st.session_state.feedback.values() if f == 'negative')
            st.metric("Positive Feedback", f"{positive} 👍")
            st.metric("Negative Feedback", f"{negative} 👎")
    
    st.markdown("---")
    
    st.markdown("## ⚙️ Actions")
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.prompt_history = []
        st.session_state.current_df = None
        st.session_state.feedback = {}
        st.rerun()
    
    if st.button("🔄 Refresh App", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("## ℹ️ About")
    st.markdown("""
    **AI Data Analyzer** v1.0
    
    Built with:
    - Streamlit
    - Pandas
    - PandasAI
    - OpenAI GPT
    
    Developed with modern data science principles
    for efficient business intelligence.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <p>Built with using Streamlit, PandasAI, and OpenAI | 
        <a href='https://github.com' style='color: #3B82F6; text-decoration: none;'>GitHub</a> | 
        <a href='https://docs.streamlit.io' style='color: #3B82F6; text-decoration: none;'>Docs</a>
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)





# Error handling for session cleanup
import atexit

def cleanup_temp_files():
    """Clean up temporary files on exit"""
    if 'uploaded_files' in st.session_state:
        for file_data in st.session_state.uploaded_files.values():
            try:
                if os.path.exists(file_data['path']):
                    os.unlink(file_data['path'])
            except:
                pass

atexit.register(cleanup_temp_files)
        
