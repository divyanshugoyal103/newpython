import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import io
import zipfile
import warnings
import re
import base64
from collections import Counter
import sqlite3
import hashlib
warnings.filterwarnings('ignore')

# Additional imports for new features
try:
    from wordcloud import WordCloud
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

try:
    import psycopg2
    import mysql.connector
    DB_CONNECTORS_AVAILABLE = True
except ImportError:
    DB_CONNECTORS_AVAILABLE = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

try:
    from fpdf import FPDF
    import pdfkit
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Initialize session state for theme and caching
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Configure Streamlit page
st.set_page_config(
    page_title="Universal Data Analyzer Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_theme_css():
    """Return CSS based on current theme"""
    if st.session_state.theme == 'dark':
        return """
        <style>
            .main-header {
                font-size: 3rem;
                color: #00d4ff;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 0 0 10px #00d4ff;
            }
            .section-header {
                font-size: 1.5rem;
                color: #ff6b6b;
                margin-top: 2rem;
                margin-bottom: 1rem;
                border-left: 4px solid #ff6b6b;
                padding-left: 1rem;
            }
            .metric-container {
                background: linear-gradient(135deg, #1e1e1e, #2d2d2d);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border: 1px solid #444;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .stApp {
                background: linear-gradient(135deg, #0c0c0c, #1a1a1a);
            }
            .nlp-insight {
                background: linear-gradient(45deg, #2d1b69, #11998e);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
            }
        </style>
        """
    else:
        return """
        <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            .section-header {
                font-size: 1.5rem;
                color: #ff7f0e;
                margin-top: 2rem;
                margin-bottom: 1rem;
                border-left: 4px solid #ff7f0e;
                padding-left: 1rem;
            }
            .metric-container {
                background: linear-gradient(135deg, #f0f2f6, #ffffff);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .nlp-insight {
                background: linear-gradient(45deg, #667eea, #764ba2);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
            }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

class UniversalDataAnalyzerPro:
    def __init__(self):
        self.data = None
        self.file_type = None
        self.file_name = None
        self.file_hash = None
        
    def get_file_hash(self, file_content):
        """Generate hash for file caching"""
        return hashlib.md5(file_content).hexdigest()
    
    def detect_file_type(self, file):
        """Detect file type based on extension and content"""
        if file.name.endswith(('.csv', '.tsv')):
            return 'csv'
        elif file.name.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif file.name.endswith('.json'):
            return 'json'
        elif file.name.endswith('.xml'):
            return 'xml'
        elif file.name.endswith('.txt'):
            return 'text'
        elif file.name.endswith('.parquet'):
            return 'parquet'
        elif file.name.endswith('.zip'):
            return 'zip'
        else:
            return 'unknown'
    
    def enhanced_date_detection(self, df):
        """Enhanced date column detection using regex and column names"""
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{2}-\d{2}-\d{4}\b',  # MM-DD-YYYY
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # M/D/YY or MM/DD/YYYY
        ]
        
        date_keywords = ['date', 'time', 'created', 'updated', 'modified', 'timestamp', 
                        'birth', 'start', 'end', 'due', 'expire', 'publish']
        
        potential_date_cols = []
        
        for col in df.columns:
            # Check column name
            if any(keyword in col.lower() for keyword in date_keywords):
                potential_date_cols.append(col)
                continue
            
            # Check content pattern
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(100)
                for pattern in date_patterns:
                    if sample_values.str.contains(pattern, regex=True).any():
                        potential_date_cols.append(col)
                        break
        
        # Convert detected date columns
        for col in potential_date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                st.info(f"🗓️ Detected and converted '{col}' to datetime")
            except:
                pass
        
        return df
    
    def load_data(self, file):
        """Load data based on file type with caching"""
        self.file_name = file.name
        self.file_type = self.detect_file_type(file)
        
        # Create file hash for caching
        file_content = file.read()
        self.file_hash = self.get_file_hash(file_content)
        
        # Check cache first
        if self.file_hash in st.session_state.cached_data:
            st.info("📦 Loading from cache...")
            cached_result = st.session_state.cached_data[self.file_hash]
            self.data = cached_result['data']
            self.file_type = cached_result['file_type']
            return True
        
        file.seek(0)  # Reset file pointer
        
        try:
            if self.file_type == 'csv':
                # Try different separators and encodings
                separators = [',', ';', '\t', '|']
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            file.seek(0)
                            self.data = pd.read_csv(file, sep=sep, encoding=encoding, 
                                                  low_memory=False)
                            if len(self.data.columns) > 1:
                                break
                        except:
                            continue
                    if self.data is not None and len(self.data.columns) > 1:
                        break
                        
            elif self.file_type == 'excel':
                file.seek(0)
                excel_file = pd.ExcelFile(file)
                if len(excel_file.sheet_names) == 1:
                    self.data = pd.read_excel(file)
                else:
                    # Handle multiple sheets
                    sheet_data = {}
                    for sheet in excel_file.sheet_names:
                        sheet_data[sheet] = pd.read_excel(file, sheet_name=sheet)
                    self.data = sheet_data
                    
            elif self.file_type == 'json':
                file.seek(0)
                json_data = json.load(file)
                if isinstance(json_data, list):
                    self.data = pd.json_normalize(json_data)
                elif isinstance(json_data, dict):
                    self.data = pd.json_normalize([json_data])
                else:
                    self.data = pd.DataFrame({'data': [json_data]})
                    
            elif self.file_type == 'xml':
                file.seek(0)
                tree = ET.parse(file)
                root = tree.getroot()
                data_list = []
                for child in root:
                    data_dict = {}
                    for subchild in child:
                        data_dict[subchild.tag] = subchild.text
                    data_list.append(data_dict)
                self.data = pd.DataFrame(data_list)
                
            elif self.file_type == 'parquet':
                file.seek(0)
                self.data = pd.read_parquet(file)
                
            elif self.file_type == 'text':
                file.seek(0)
                content = file.read().decode('utf-8')
                lines = content.split('\n')
                self.data = pd.DataFrame({'text': lines})
                
            elif self.file_type == 'zip':
                file.seek(0)
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    st.info(f"ZIP file contains: {', '.join(file_list)}")
                    # Extract and analyze first CSV/Excel file found
                    for filename in file_list:
                        if filename.endswith(('.csv', '.xlsx', '.json')):
                            with zip_ref.open(filename) as extracted_file:
                                if filename.endswith('.csv'):
                                    self.data = pd.read_csv(extracted_file)
                                elif filename.endswith('.xlsx'):
                                    self.data = pd.read_excel(extracted_file)
                                elif filename.endswith('.json'):
                                    json_data = json.load(extracted_file)
                                    self.data = pd.json_normalize(json_data)
                                break
            
            # Apply enhanced date detection
            if self.data is not None and not isinstance(self.data, dict):
                self.data = self.enhanced_date_detection(self.data)
            elif isinstance(self.data, dict):
                for sheet_name, df in self.data.items():
                    self.data[sheet_name] = self.enhanced_date_detection(df)
            
            # Cache the result
            st.session_state.cached_data[self.file_hash] = {
                'data': self.data,
                'file_type': self.file_type,
                'timestamp': datetime.now()
            }
            
            return True
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def nlp_analysis(self, df):
        """Perform NLP analysis on text data"""
        st.markdown('<div class="section-header">📝 NLP Text Analysis</div>', unsafe_allow_html=True)
        
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains substantial text (avg length > 10 chars)
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 10:
                    text_cols.append(col)
        
        if not text_cols:
            st.info("No substantial text columns found for NLP analysis.")
            return
        
        selected_text_col = st.selectbox("Select text column for NLP analysis:", text_cols)
        
        if selected_text_col:
            text_data = df[selected_text_col].dropna().astype(str)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic text statistics
                st.subheader("📊 Text Statistics")
                total_words = text_data.str.split().str.len().sum()
                avg_words = text_data.str.split().str.len().mean()
                total_chars = text_data.str.len().sum()
                avg_chars = text_data.str.len().mean()
                
                st.metric("Total Words", f"{total_words:,}")
                st.metric("Average Words per Entry", f"{avg_words:.1f}")
                st.metric("Total Characters", f"{total_chars:,}")
                st.metric("Average Characters per Entry", f"{avg_chars:.1f}")
            
            with col2:
                # Word frequency analysis
                st.subheader("🔤 Most Frequent Words")
                
                if NLTK_AVAILABLE:
                    try:
                        # Combine all text
                        all_text = ' '.join(text_data.values).lower()
                        
                        # Tokenize and remove stopwords
                        tokens = word_tokenize(all_text)
                        stop_words = set(stopwords.words('english'))
                        filtered_tokens = [word for word in tokens if word.isalnum() and 
                                         word not in stop_words and len(word) > 2]
                        
                        # Get top words
                        word_freq = Counter(filtered_tokens)
                        top_words = word_freq.most_common(10)
                        
                        # Display as dataframe
                        word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                        st.dataframe(word_df, use_container_width=True)
                        
                        # Word frequency bar chart
                        fig = px.bar(word_df, x='Word', y='Frequency', 
                                   title='Top 10 Most Frequent Words')
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"NLTK analysis failed: {str(e)}")
                else:
                    # Simple word count without NLTK
                    all_text = ' '.join(text_data.values).lower()
                    words = re.findall(r'\b\w+\b', all_text)
                    word_freq = Counter([w for w in words if len(w) > 2])
                    top_words = word_freq.most_common(10)
                    
                    word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                    st.dataframe(word_df, use_container_width=True)
            
            # Word cloud
            try:
                st.subheader("☁️ Word Cloud")
                if NLTK_AVAILABLE:
                    wordcloud_text = ' '.join(filtered_tokens)
                else:
                    wordcloud_text = all_text
                
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white' if st.session_state.theme == 'light' else 'black',
                                    colormap='viridis').generate(wordcloud_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
            except Exception as e:
                st.info("Word cloud generation requires additional dependencies.")
            
            # Text length distribution
            st.subheader("📏 Text Length Distribution")
            text_lengths = text_data.str.len()
            
            fig = px.histogram(x=text_lengths, nbins=30, 
                             title='Distribution of Text Lengths',
                             labels={'x': 'Text Length (characters)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
    
    def interactive_pivot_table(self, df):
        """Create interactive pivot table"""
        st.markdown('<div class="section-header">🔄 Interactive Pivot Analysis</div>', unsafe_allow_html=True)
        
        if AGGRID_AVAILABLE:
            st.subheader("Advanced Grid View with AgGrid")
            
            # Configure grid options
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_selection('multiple', use_checkbox=True)
            gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
            gridOptions = gb.build()
            
            # Display grid
            grid_response = AgGrid(
                df,
                gridOptions=gridOptions,
                data_return_mode='AS_INPUT',
                update_mode='MODEL_CHANGED',
                fit_columns_on_grid_load=False,
                enable_enterprise_modules=True,
                height=400,
                reload_data=False
            )
        
        # Fallback: Simple pivot interface
        st.subheader("Simple Pivot Table Builder")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            index_col = st.selectbox("Row (Index):", ['None'] + categorical_cols)
        with col2:
            columns_col = st.selectbox("Columns:", ['None'] + categorical_cols)
        with col3:
            values_col = st.selectbox("Values:", numeric_cols)
        
        if index_col != 'None' and values_col:
            try:
                pivot_kwargs = {
                    'index': index_col,
                    'values': values_col,
                    'aggfunc': st.selectbox("Aggregation:", ['mean', 'sum', 'count', 'min', 'max'])
                }
                
                if columns_col != 'None':
                    pivot_kwargs['columns'] = columns_col
                
                pivot_table = df.pivot_table(**pivot_kwargs)
                
                st.subheader("Pivot Table Result")
                st.dataframe(pivot_table, use_container_width=True)
                
                # Visualize pivot table
                if pivot_table.shape[0] <= 20 and pivot_table.shape[1] <= 20:
                    fig = px.imshow(pivot_table, 
                                  title=f"Pivot Table Heatmap: {values_col} by {index_col}" + 
                                        (f" and {columns_col}" if columns_col != 'None' else ""),
                                  aspect='auto')
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating pivot table: {str(e)}")
    
    def generate_report(self, df):
        """Generate downloadable analysis report"""
        st.markdown('<div class="section-header">📄 Analysis Report Generator</div>', unsafe_allow_html=True)
        
        report_format = st.selectbox("Select report format:", ["Markdown", "HTML", "PDF"])
        
        if st.button("Generate Report"):
            # Create comprehensive report
            report_content = self._create_report_content(df)
            
            if report_format == "Markdown":
                st.download_button(
                    label="📥 Download Markdown Report",
                    data=report_content,
                    file_name=f"analysis_report_{self.file_name.split('.')[0]}.md",
                    mime="text/markdown"
                )
            
            elif report_format == "HTML":
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Data Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1, h2, h3 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        .metric {{ background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    {report_content.replace('\n', '<br>').replace('##', '<h2>').replace('#', '<h1>')}
                </body>
                </html>
                """
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_content,
                    file_name=f"analysis_report_{self.file_name.split('.')[0]}.html",
                    mime="text/html"
                )
            
            elif report_format == "PDF" and PDF_AVAILABLE:
                try:
                    # Create PDF (simplified version)
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    # Add content (simplified for PDF)
                    pdf.cell(200, 10, txt="Data Analysis Report", ln=1, align='C')
                    pdf.cell(200, 10, txt=f"File: {self.file_name}", ln=1)
                    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
                    
                    pdf_output = pdf.output(dest='S').encode('latin-1')
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_output,
                        file_name=f"analysis_report_{self.file_name.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
                    st.info("Consider using HTML or Markdown format instead.")
    
    def _create_report_content(self, df):
        """Create comprehensive report content"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        report = f"""# Data Analysis Report
        
## File Information
- **Filename**: {self.file_name}
- **File Type**: {self.file_type.upper()}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Rows**: {len(df):,}
- **Columns**: {len(df.columns)}
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **Missing Data**: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%

## Column Information
- **Numeric Columns**: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})
- **Categorical Columns**: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})

## Statistical Summary
"""
        
        if len(numeric_cols) > 0:
            report += "\n### Numeric Columns Statistics\n"
            desc_stats = df[numeric_cols].describe()
            report += desc_stats.to_string()
        
        if len(categorical_cols) > 0:
            report += "\n\n### Categorical Columns Summary\n"
            for col in categorical_cols[:3]:  # Limit to first 3
                unique_count = df[col].nunique()
                most_common = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
                report += f"- **{col}**: {unique_count} unique values, most common: '{most_common}'\n"
        
        report += f"\n## Key Insights\n"
        report += f"- Dataset contains {len(df)} records across {len(df.columns)} variables\n"
        report += f"- {len(numeric_cols)} numeric and {len(categorical_cols)} categorical variables identified\n"
        
        if len(numeric_cols) > 1:
            # Add correlation insights
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                report += f"- Found {len(high_corr_pairs)} highly correlated variable pairs (|r| > 0.7)\n"
        
        return report
    
    def database_connector(self):
        """Connect to external databases"""
        if not DB_CONNECTORS_AVAILABLE:
            st.info("Database connectors require additional packages: `pip install psycopg2 mysql-connector-python`")
            return
        
        st.markdown('<div class="section-header">🗄️ Database Connection</div>', unsafe_allow_html=True)
        
        db_type = st.selectbox("Database Type:", ["PostgreSQL", "MySQL", "SQLite"])
        
        if db_type == "SQLite":
            uploaded_db = st.file_uploader("Upload SQLite Database", type=['db', 'sqlite', 'sqlite3'])
            if uploaded_db:
                with open("temp.db", "wb") as f:
                    f.write(uploaded_db.read())
                
                conn = sqlite3.connect("temp.db")
                
                # Get table names
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                
                if not tables.empty:
                    selected_table = st.selectbox("Select Table:", tables['name'].tolist())
                    
                    if st.button("Load Table"):
                        self.data = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                        st.success(f"Loaded {len(self.data)} rows from {selected_table}")
                
                conn.close()
        
        else:
            # Connection parameters
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host:", "localhost")
                database = st.text_input("Database:")
            with col2:
                username = st.text_input("Username:")
                password = st.text_input("Password:", type="password")
            
            port = st.number_input("Port:", value=5432 if db_type == "PostgreSQL" else 3306)
            
            if st.button("Test Connection"):
                try:
                    if db_type == "PostgreSQL":
                        conn = psycopg2.connect(
                            host=host, database=database, 
                            user=username, password=password, port=port
                        )
                    else:  # MySQL
                        conn = mysql.connector.connect(
                            host=host, database=database,
                            user=username, password=password, port=port
                        )
                    
                    st.success("✅ Connection successful!")
                    
                    # Custom SQL query
                    query = st.text_area("SQL Query:", "SELECT * FROM your_table LIMIT 100")
                    
                    if st.button("Execute Query"):
                        self.data = pd.read_sql_query(query, conn)
                        st.success(f"Query executed successfully! Loaded {len(self.data)} rows.")
                    
                    conn.close()
                    
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
    
    def basic_info(self):
        """Display basic information about the dataset"""
        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):  # Multiple Excel sheets
            for sheet_name, df in self.data.items():
                st.subheader(f"Sheet: {sheet_name}")
                self._display_basic_info(df)
        else:
            self._display_basic_info(self.data)
    
    def _display_basic_info(self, df):
        """Helper function to display basic info for a dataframe"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Display column information
        st.subheader("Column Information")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Missing %': (df.isnull().mean() * 100).round(2),
            'Unique Values': df.nunique()
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Show detailed statistics
        st.subheader("Detailed Statistics")
        tab1, tab2 = st.tabs(["Numeric Columns", "Categorical Columns"])
        
        with tab1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
            else:
                st.info("No numeric columns found")
        
        with tab2:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cat_stats = []
                for col in categorical_cols:
                    cat_stats.append({
                        'Column': col,
                        'Unique Values': df[col].nunique(),
                        'Most Common': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                        'Frequency': df[col].value_counts().iloc[0] if not df[col].mode().empty else 0
                    })
                st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)
            else:
                st.info("No categorical columns found")

    def data_cleaning(self):
        """Interactive data cleaning interface"""
        st.markdown('<div class="section-header">🧹 Data Cleaning Tools</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):  # Multiple sheets
            sheet_names = list(self.data.keys())
            selected_sheet = st.selectbox("Select Sheet:", sheet_names)
            df = self.data[selected_sheet]
        else:
            df = self.data
        
        st.subheader("Missing Data Handling")
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_action = st.selectbox(
                "Action for missing values:",
                ["Show summary", "Drop rows", "Drop columns", "Fill with value"]
            )
        
        with col2:
            if missing_action == "Fill with value":
                fill_value = st.text_input("Fill value:", "0")
                if df.select_dtypes(include=[np.number]).columns.any():
                    fill_value = float(fill_value) if '.' in fill_value else int(fill_value)
        
        if st.button("Apply Missing Value Action"):
            if missing_action == "Drop rows":
                df = df.dropna()
                st.success(f"Removed rows with missing values. New shape: {df.shape}")
            elif missing_action == "Drop columns":
                df = df.dropna(axis=1)
                st.success(f"Removed columns with missing values. New shape: {df.shape}")
            elif missing_action == "Fill with value":
                df = df.fillna(fill_value)
                st.success(f"Filled missing values with {fill_value}")
            
            # Update data
            if isinstance(self.data, dict):
                self.data[selected_sheet] = df
            else:
                self.data = df
        
        st.subheader("Column Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            column_action = st.selectbox(
                "Column operation:",
                ["Rename", "Drop", "Change type", "Extract text"]
            )
        
        with col2:
            selected_column = st.selectbox("Select column:", df.columns)
        
        if column_action == "Rename":
            new_name = st.text_input("New column name:")
            if st.button("Rename Column"):
                df = df.rename(columns={selected_column: new_name})
                st.success(f"Renamed '{selected_column}' to '{new_name}'")
        
        elif column_action == "Drop":
            if st.button("Drop Column"):
                df = df.drop(columns=[selected_column])
                st.success(f"Dropped column '{selected_column}'")
        
        elif column_action == "Change type":
            new_type = st.selectbox("New type:", ["string", "numeric", "datetime", "category"])
            if st.button("Change Type"):
                try:
                    if new_type == "numeric":
                        df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce')
                    elif new_type == "datetime":
                        df[selected_column] = pd.to_datetime(df[selected_column], errors='coerce')
                    elif new_type == "category":
                        df[selected_column] = df[selected_column].astype('category')
                    else:  # string
                        df[selected_column] = df[selected_column].astype(str)
                    st.success(f"Changed '{selected_column}' to {new_type}")
                except Exception as e:
                    st.error(f"Error converting column: {str(e)}")
        
        elif column_action == "Extract text":
            pattern = st.text_input("Regex pattern:")
            if st.button("Extract"):
                try:
                    extracted = df[selected_column].str.extract(pattern)
                    df = pd.concat([df, extracted], axis=1)
                    st.success(f"Added {extracted.shape[1]} new columns from extraction")
                except Exception as e:
                    st.error(f"Extraction failed: {str(e)}")
        
        # Update data after column operations
        if isinstance(self.data, dict):
            self.data[selected_sheet] = df
        else:
            self.data = df
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):  # Multiple sheets
            sheet_names = list(self.data.keys())
            selected_sheet = st.selectbox("Select Sheet for Analysis:", sheet_names)
            df = self.data[selected_sheet]
        else:
            df = self.data
        
        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("📊 Numeric Variable Distributions")
            
            selected_num_col = st.selectbox("Select numeric column:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_num_col, nbins=30, 
                                 title=f"Distribution of {selected_num_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=selected_num_col, 
                           title=f"Box Plot of {selected_num_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Q-Q plot
            st.subheader("📈 Q-Q Plot (Normality Check)")
            qq_data = df[selected_num_col].dropna()
            qq_data = (qq_data - qq_data.mean()) / qq_data.std()  # Standardize
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(qq_data)))),
                y=np.sort(qq_data),
                mode='markers',
                name='Data'
            ))
            fig.add_trace(go.Scatter(
                x=[-3, 3],
                y=[-3, 3],
                mode='lines',
                name='Normal Reference',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Q-Q Plot for {selected_num_col}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.subheader("📊 Categorical Variable Analysis")
            
            selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
            
            # Value counts
            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']
            
            # Bar chart
            fig = px.bar(value_counts.head(20), 
                         x='Value', y='Count',
                         title=f"Value Counts for {selected_cat_col}")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie chart for top categories
            top_n = st.slider("Show top N categories:", 3, 20, 5)
            top_categories = value_counts.head(top_n)
            
            fig = px.pie(top_categories, names='Value', values='Count',
                        title=f"Top {top_n} Categories in {selected_cat_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.subheader("📈 Correlation Analysis")
            
            corr_method = st.selectbox("Correlation method:", 
                                     ['pearson', 'spearman', 'kendall'])
            
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr(method=corr_method)
            
            fig = px.imshow(corr_matrix,
                          text_auto=True,
                          aspect="auto",
                          title=f"Correlation Matrix ({corr_method.capitalize()})",
                          color_continuous_scale='RdBu_r',
                          range_color=[-1, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Find high correlations
            st.subheader("🔍 High Correlations")
            
            threshold = st.slider("Correlation threshold:", 0.5, 0.95, 0.7, 0.05)
            
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        high_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                high_corr_df = pd.DataFrame(high_corr)
                st.dataframe(high_corr_df.sort_values('Correlation', ascending=False), 
                            use_container_width=True)
                
                # Scatter plot of highest correlation pair
                if len(high_corr) > 0:
                    highest = high_corr_df.iloc[0]
                    fig = px.scatter(df, 
                                    x=highest['Variable 1'], 
                                    y=highest['Variable 2'],
                                    title=f"Highest Correlation: {highest['Variable 1']} vs {highest['Variable 2']}",
                                    trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No variable pairs with |correlation| > {threshold}")
    
    def advanced_analysis(self):
        """Perform advanced statistical analysis"""
        st.markdown('<div class="section-header">🧪 Advanced Analysis</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):  # Multiple sheets
            sheet_names = list(self.data.keys())
            selected_sheet = st.selectbox("Select Sheet for Advanced Analysis:", sheet_names)
            df = self.data[selected_sheet]
        else:
            df = self.data
        
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["PCA (Dimensionality Reduction)", "Clustering (K-Means)", 
             "Outlier Detection", "Time Series Analysis"]
        )
        
        if analysis_type == "PCA (Dimensionality Reduction)":
            self._pca_analysis(df)
        elif analysis_type == "Clustering (K-Means)":
            self._kmeans_clustering(df)
        elif analysis_type == "Outlier Detection":
            self._outlier_detection(df)
        elif analysis_type == "Time Series Analysis":
            self._time_series_analysis(df)
    
    def _pca_analysis(self, df):
        """Perform Principal Component Analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.warning("PCA requires at least 2 numeric columns")
            return
        
        st.subheader("Principal Component Analysis")
        
        # Select columns for PCA
        selected_cols = st.multiselect(
            "Select numeric columns for PCA:",
            numeric_cols,
            default=numeric_cols.tolist()[:5]
        )
        
        if len(selected_cols) < 2:
            st.warning("Please select at least 2 columns")
            return
        
        # Standardize data
        X = df[selected_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        n_components = st.slider("Number of components:", 2, min(10, len(selected_cols)), 2)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # Explained variance
        st.subheader("Explained Variance")
        exp_var = pca.explained_variance_ratio_
        
        fig = px.bar(x=[f"PC{i+1}" for i in range(n_components)], 
                    y=exp_var,
                    labels={'x': 'Principal Component', 'y': 'Explained Variance'},
                    title="Variance Explained by Each Principal Component")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative explained variance
        cum_exp_var = np.cumsum(exp_var)
        fig = px.line(x=[f"PC{i+1}" for i in range(n_components)], 
                     y=cum_exp_var,
                     labels={'x': 'Principal Component', 'y': 'Cumulative Explained Variance'},
                     title="Cumulative Explained Variance")
        fig.update_yaxes(range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)
        
        # PCA loadings
        st.subheader("Component Loadings")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=selected_cols
        )
        st.dataframe(loadings.style.background_gradient(cmap='RdBu_r', axis=None, vmin=-1, vmax=1), 
                    use_container_width=True)
        
        # Biplot for first two components
        if n_components >= 2:
            st.subheader("Biplot (First Two Components)")
            
            # Create biplot
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=principal_components[:, 0],
                y=principal_components[:, 1],
                mode='markers',
                name='Data Points',
                marker=dict(size=8, opacity=0.5)
            ))
            
            # Add variable vectors
            for i, var in enumerate(selected_cols):
                fig.add_trace(go.Scatter(
                    x=[0, pca.components_[0, i] * 3],
                    y=[0, pca.components_[1, i] * 3],
                    mode='lines+text',
                    name=var,
                    line=dict(width=2),
                    text=[None, var],
                    textposition="top center"
                ))
            
            fig.update_layout(
                title="Biplot of PC1 vs PC2",
                xaxis_title=f"PC1 ({exp_var[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({exp_var[1]*100:.1f}%)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _kmeans_clustering(self, df):
        """Perform K-Means clustering"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.warning("Clustering requires at least 2 numeric columns")
            return
        
        st.subheader("K-Means Clustering")
        
        # Select columns for clustering
        selected_cols = st.multiselect(
            "Select numeric columns for clustering:",
            numeric_cols,
            default=numeric_cols.tolist()[:2]
        )
        
        if len(selected_cols) < 2:
            st.warning("Please select at least 2 columns")
            return
        
        # Standardize data
        X = df[selected_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal k using elbow method
        st.subheader("Determine Optimal Number of Clusters")
        
        max_clusters = min(10, len(X_scaled))
        distortions = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            distortions.append(kmeans.inertia_)
        
        fig = px.line(x=list(range(1, max_clusters + 1)), y=distortions,
                     labels={'x': 'Number of clusters', 'y': 'Distortion'},
                     title='Elbow Method for Optimal k')
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
        
        # Select number of clusters
        n_clusters = st.slider("Number of clusters:", 2, max_clusters, 3)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to original data
        X_clustered = X.copy()
        X_clustered['Cluster'] = clusters
        
        # Plot clusters
        if len(selected_cols) == 2:
            fig = px.scatter(X_clustered, 
                           x=selected_cols[0], 
                           y=selected_cols[1],
                           color='Cluster',
                           title=f"K-Means Clustering (k={n_clusters})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # For >2 dimensions, show first two components
            st.warning("Showing first two dimensions of the data")
            fig = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1],
                           color=clusters,
                           title=f"K-Means Clustering (k={n_clusters})")
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        st.subheader("Cluster Statistics")
        X_clustered['Cluster'] = X_clustered['Cluster'].astype(str)
        cluster_stats = X_clustered.groupby('Cluster').describe().T
        st.dataframe(cluster_stats, use_container_width=True)
    
    def _outlier_detection(self, df):
        """Detect outliers using various methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("Outlier detection requires numeric columns")
            return
        
        st.subheader("Outlier Detection")
        
        method = st.selectbox(
            "Detection method:",
            ["Z-Score", "IQR (Boxplot)", "Isolation Forest"]
        )
        
        selected_col = st.selectbox("Select numeric column:", numeric_cols)
        
        X = df[selected_col].dropna().values.reshape(-1, 1)
        
        if method == "Z-Score":
            threshold = st.slider("Z-Score threshold:", 2.0, 5.0, 3.0, 0.5)
            
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(X))
            outliers = (z_scores > threshold).flatten()
            
            st.metric("Outliers detected", f"{outliers.sum()} ({outliers.mean()*100:.1f}%)")
            
            # Show outliers
            fig = px.scatter(x=range(len(X)), y=X.flatten(),
                            color=outliers,
                            title=f"Outlier Detection (Z-Score > {threshold})",
                            labels={'x': 'Index', 'y': selected_col})
            st.plotly_chart(fig, use_container_width=True)
        
        elif method == "IQR (Boxplot)":
            # Show boxplot
            fig = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate IQR bounds
            q1 = np.percentile(X, 25)
            q3 = np.percentile(X, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (X < lower_bound) | (X > upper_bound)
            st.metric("Outliers detected", f"{outliers.sum()} ({outliers.mean()*100:.1f}%)")
        
        elif method == "Isolation Forest":
            contamination = st.slider("Expected outlier fraction:", 0.01, 0.5, 0.05, 0.01)
            
            clf = IsolationForest(contamination=contamination, random_state=42)
            preds = clf.fit_predict(X)
            outliers = preds == -1
            
            st.metric("Outliers detected", f"{outliers.sum()} ({outliers.mean()*100:.1f}%)")
            
            # Show outliers
            fig = px.scatter(x=range(len(X)), y=X.flatten(),
                            color=outliers,
                            title=f"Outlier Detection (Isolation Forest)",
                            labels={'x': 'Index', 'y': selected_col})
            st.plotly_chart(fig, use_container_width=True)
    
    def _time_series_analysis(self, df):
        """Perform time series analysis"""
        date_cols = df.select_dtypes(include=['datetime']).columns
        
        if len(date_cols) == 0:
            st.warning("No datetime columns found for time series analysis")
            return
        
        st.subheader("Time Series Analysis")
        
        date_col = st.selectbox("Select datetime column:", date_cols)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for time series analysis")
            return
        
        value_col = st.selectbox("Select numeric column to analyze:", numeric_cols)
        
        # Resample time series
        st.subheader("Time Series Aggregation")
        
        resample_freq = st.selectbox(
            "Aggregation frequency:",
            ["D (Daily)", "W (Weekly)", "M (Monthly)", "Q (Quarterly)", "Y (Yearly)"]
        )
        
        freq_map = {
            "D (Daily)": "D",
            "W (Weekly)": "W",
            "M (Monthly)": "M",
            "Q (Quarterly)": "Q",
            "Y (Yearly)": "Y"
        }
        
        agg_func = st.selectbox(
            "Aggregation function:",
            ["mean", "sum", "median", "min", "max"]
        )
        
        # Create time series
        ts_df = df.set_index(date_col)[value_col].dropna()
        ts_resampled = ts_df.resample(freq_map[resample_freq]).agg(agg_func)
        
        # Plot time series
        fig = px.line(ts_resampled, 
                     title=f"{value_col} over Time ({resample_freq})",
                     labels={'value': value_col})
        st.plotly_chart(fig, use_container_width=True)
        
        # Decomposition
        st.subheader("Time Series Decomposition")
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(ts_resampled.fillna(ts_resampled.mean()), 
                                            model='additive', period=12)
            
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                              subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
            
            fig.add_trace(go.Scatter(x=decomposition.observed.index, 
                                   y=decomposition.observed,
                                   name="Observed"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=decomposition.trend.index, 
                                   y=decomposition.trend,
                                   name="Trend"), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=decomposition.seasonal.index, 
                                   y=decomposition.seasonal,
                                   name="Seasonal"), row=3, col=1)
            
            fig.add_trace(go.Scatter(x=decomposition.resid.index, 
                                   y=decomposition.resid,
                                   name="Residual"), row=4, col=1)
            
            fig.update_layout(height=800, title_text="Time Series Decomposition")
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.info("Time series decomposition requires statsmodels: `pip install statsmodels`")
        except Exception as e:
            st.error(f"Decomposition failed: {str(e)}")
    
    def save_analysis(self):
        """Save current analysis to history"""
        if self.data is None:
            return
        
        analysis_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'file_name': self.file_name,
            'data_shape': self.data.shape if not isinstance(self.data, dict) else {k: v.shape for k, v in self.data.items()},
            'file_hash': self.file_hash
        }
        
        st.session_state.analysis_history.append(analysis_entry)
        st.success("Analysis saved to history!")
    
    def show_history(self):
        """Display analysis history"""
        st.markdown('<div class="section-header">🕰️ Analysis History</div>', unsafe_allow_html=True)
        
        if not st.session_state.analysis_history:
            st.info("No analysis history yet")
            return
        
        history_df = pd.DataFrame(st.session_state.analysis_history)
        
        # Display history table
        st.dataframe(history_df, use_container_width=True)
        
        # Option to load from history
        selected_index = st.selectbox(
            "Select analysis to load:",
            range(len(st.session_state.analysis_history)),
            format_func=lambda x: f"{st.session_state.analysis_history[x]['timestamp']} - {st.session_state.analysis_history[x]['file_name']}"
        )
        
        if st.button("Load Selected Analysis"):
            selected_entry = st.session_state.analysis_history[selected_index]
            cached_data = st.session_state.cached_data.get(selected_entry['file_hash'])
            
            if cached_data:
                self.data = cached_data['data']
                self.file_name = selected_entry['file_name']
                self.file_hash = selected_entry['file_hash']
                self.file_type = cached_data['file_type']
                st.success(f"Loaded analysis from {selected_entry['timestamp']}")
            else:
                st.error("Cached data not found for this analysis")

# Main application function
def main():
    """Main application function"""
    st.markdown('<div class="main-header">Universal Data Analyzer Pro</div>', unsafe_allow_html=True)
    
    # Theme toggle
    if st.sidebar.button(f"Switch to {'Dark' if st.session_state.theme == 'light' else 'Light'} Theme"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.experimental_rerun()
    
    # Initialize analyzer
    analyzer = UniversalDataAnalyzerPro()
    
    # File upload
    st.sidebar.markdown('<div class="section-header">📂 Data Input</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Upload your data file", 
        type=['csv', 'xlsx', 'json', 'xml', 'txt', 'parquet', 'zip']
    )
    
    # Database connection option
    if st.sidebar.checkbox("Connect to Database"):
        analyzer.database_connector()
    
    # Load data if file uploaded
    if uploaded_file:
        if analyzer.load_data(uploaded_file):
            st.sidebar.success(f"Successfully loaded {uploaded_file.name}")
            
            # Show basic info
            analyzer.basic_info()
            
            # Analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🧹 Data Cleaning", "🔍 EDA", "🧪 Advanced", "📝 NLP", "📄 Report"
            ])
            
            with tab1:
                analyzer.data_cleaning()
            
            with tab2:
                analyzer.exploratory_analysis()
            
            with tab3:
                analyzer.advanced_analysis()
            
            with tab4:
                if isinstance(analyzer.data, dict):
                    st.warning("NLP analysis not available for multi-sheet data")
                else:
                    analyzer.nlp_analysis(analyzer.data)
            
            with tab5:
                if isinstance(analyzer.data, dict):
                    st.warning("Report generation not available for multi-sheet data")
                else:
                    analyzer.generate_report(analyzer.data)
            
            # Save analysis button
            if st.sidebar.button("💾 Save Current Analysis"):
                analyzer.save_analysis()
            
            # Show history
            if st.sidebar.checkbox("Show Analysis History"):
                analyzer.show_history()
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About Universal Data Analyzer Pro
    A comprehensive tool for data analysis, visualization, and reporting.
    
    **Features:**
    - Load multiple file formats
    - Interactive data cleaning
    - Exploratory analysis
    - Advanced statistical methods
    - NLP capabilities
    - Report generation
    
    Created with ❤️ using Streamlit
    """)

if __name__ == "__main__":
    main()
