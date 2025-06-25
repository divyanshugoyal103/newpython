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
        
               def _display_basic_info(self, df):
        """Helper function to display basic info for a dataframe"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("📊 Total Rows", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("📋 Total Columns", len(df.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("💾 Memory Usage", f"{memory_usage:.2f} MB")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("❓ Missing Data", f"{missing_percentage:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data types summary
        st.subheader("📊 Column Data Types")
        dtype_counts = df.dtypes.value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(dtype_counts.reset_index().rename(columns={'index': 'Data Type', 0: 'Count'}))
        
        with col2:
            fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, 
                        title="Distribution of Data Types")
            st.plotly_chart(fig, use_container_width=True)
        
        # Column details
        st.subheader("🔍 Column Details")
        column_info = []
        for col in df.columns:
            col_info = {
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': df[col].count(),
                'Null Count': df[col].isnull().sum(),
                'Null %': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
                'Unique Values': df[col].nunique(),
                'Sample Value': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
            }
            column_info.append(col_info)
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)
        
        # Sample data preview
        st.subheader("👀 Data Preview")
        preview_rows = st.slider("Number of rows to preview:", min_value=5, max_value=min(100, len(df)), value=10)
        st.dataframe(df.head(preview_rows), use_container_width=True)
    
    def statistical_analysis(self):
        """Perform statistical analysis"""
        st.markdown('<div class="section-header">📈 Statistical Analysis</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            sheet_name = st.selectbox("Select sheet for analysis:", list(self.data.keys()))
            df = self.data[sheet_name]
        else:
            df = self.data
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for statistical analysis.")
            return
        
        # Descriptive statistics
        st.subheader("📊 Descriptive Statistics")
        desc_stats = df[numeric_cols].describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Distribution analysis
        st.subheader("📈 Distribution Analysis")
        selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_col, nbins=30, 
                                 title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical tests
            st.subheader("🧪 Normality Tests")
            col1, col2, col3 = st.columns(3)
            
            data_clean = df[selected_col].dropna()
            
            with col1:
                # Shapiro-Wilk test (for small samples)
                if len(data_clean) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(data_clean)
                    st.metric("Shapiro-Wilk p-value", f"{shapiro_p:.6f}")
                    st.caption("Normal if p > 0.05")
                else:
                    st.info("Sample too large for Shapiro-Wilk test")
            
            with col2:
                # Jarque-Bera test
                jb_stat, jb_p = stats.jarque_bera(data_clean)
                st.metric("Jarque-Bera p-value", f"{jb_p:.6f}")
                st.caption("Normal if p > 0.05")
            
            with col3:
                # Anderson-Darling test
                ad_stat, ad_critical, ad_significance = stats.anderson(data_clean)
                st.metric("Anderson-Darling Statistic", f"{ad_stat:.4f}")
                st.caption(f"Critical value (5%): {ad_critical[2]:.4f}")
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Analysis")
            
            corr_method = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"])
            corr_matrix = df[numeric_cols].corr(method=corr_method)
            
            # Correlation heatmap
            fig = px.imshow(corr_matrix, 
                          title=f"{corr_method.capitalize()} Correlation Matrix",
                          color_continuous_scale='RdBu',
                          aspect='auto')
            fig.update_layout(width=700, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations
            st.subheader("💪 Strong Correlations (|r| > 0.7)")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                st.dataframe(strong_corr_df, use_container_width=True)
            else:
                st.info("No strong correlations found.")
    
    def data_visualization(self):
        """Create various data visualizations"""
        st.markdown('<div class="section-header">📊 Data Visualization</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            sheet_name = st.selectbox("Select sheet for visualization:", list(self.data.keys()))
            df = self.data[sheet_name]
        else:
            df = self.data
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Chart type selection
        chart_type = st.selectbox("Select visualization type:", [
            "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
            "Box Plot", "Violin Plot", "Heatmap", "Pair Plot",
            "Time Series", "Sunburst Chart", "Treemap", "3D Scatter"
        ])
        
        if chart_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_cols)
            with col3:
                color_col = st.selectbox("Color by:", ['None'] + categorical_cols + numeric_cols)
            
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, 
                               color=color_col if color_col != 'None' else None,
                               title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Line Chart":
            if datetime_cols:
                x_col = st.selectbox("X-axis (time):", datetime_cols)
                y_col = st.selectbox("Y-axis:", numeric_cols)
                
                if x_col and y_col:
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over time")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No datetime columns found for time series visualization.")
        
        elif chart_type == "Bar Chart":
            if categorical_cols:
                x_col = st.selectbox("Category:", categorical_cols)
                y_col = st.selectbox("Value:", numeric_cols) if numeric_cols else None
                
                if x_col:
                    if y_col:
                        # Aggregate data
                        agg_data = df.groupby(x_col)[y_col].mean().reset_index()
                        fig = px.bar(agg_data, x=x_col, y=y_col, 
                                   title=f"Average {y_col} by {x_col}")
                    else:
                        # Count plot
                        value_counts = df[x_col].value_counts().reset_index()
                        fig = px.bar(value_counts, x='index', y=x_col, 
                                   title=f"Count of {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns found for bar chart.")
        
        elif chart_type == "Histogram":
            col = st.selectbox("Column:", numeric_cols)
            bins = st.slider("Number of bins:", 10, 100, 30)
            
            if col:
                fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Box Plot":
            y_col = st.selectbox("Y-axis:", numeric_cols)
            x_col = st.selectbox("Group by:", ['None'] + categorical_cols)
            
            if y_col:
                fig = px.box(df, y=y_col, x=x_col if x_col != 'None' else None,
                           title=f"Box Plot of {y_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Violin Plot":
            y_col = st.selectbox("Y-axis:", numeric_cols)
            x_col = st.selectbox("Group by:", ['None'] + categorical_cols)
            
            if y_col:
                fig = px.violin(df, y=y_col, x=x_col if x_col != 'None' else None,
                              title=f"Violin Plot of {y_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Heatmap":
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:5])
                if selected_cols:
                    corr_matrix = df[selected_cols].corr()
                    fig = px.imshow(corr_matrix, title="Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for heatmap.")
        
        elif chart_type == "Pair Plot":
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:4])
                if len(selected_cols) >= 2:
                    fig = px.scatter_matrix(df[selected_cols], title="Pair Plot")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for pair plot.")
        
        elif chart_type == "3D Scatter":
            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis:", numeric_cols)
                with col3:
                    z_col = st.selectbox("Z-axis:", numeric_cols)
                
                color_col = st.selectbox("Color by:", ['None'] + categorical_cols)
                
                if x_col and y_col and z_col:
                    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                                      color=color_col if color_col != 'None' else None,
                                      title=f"3D Scatter: {x_col}, {y_col}, {z_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 3 numeric columns for 3D scatter plot.")
    
    def advanced_analysis(self):
        """Perform advanced analysis including clustering and PCA"""
        st.markdown('<div class="section-header">🔬 Advanced Analytics</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            sheet_name = st.selectbox("Select sheet for advanced analysis:", list(self.data.keys()))
            df = self.data[sheet_name]
        else:
            df = self.data
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for advanced analysis.")
            return
        
        analysis_type = st.selectbox("Select analysis type:", [
            "Principal Component Analysis (PCA)",
            "K-Means Clustering", 
            "Hierarchical Clustering",
            "Outlier Detection",
            "Time Series Decomposition"
        ])
        
        # Data preparation
        selected_cols = st.multiselect("Select columns for analysis:", numeric_cols, default=numeric_cols[:5])
        
        if not selected_cols:
            st.warning("Please select at least one column.")
            return
        
        # Handle missing values
        df_clean = df[selected_cols].dropna()
        
        if len(df_clean) == 0:
            st.error("No valid data after removing missing values.")
            return
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_clean)
        
        if analysis_type == "Principal Component Analysis (PCA)":
            st.subheader("🎯 PCA Analysis")
            
            # Determine number of components
            n_components = min(len(selected_cols), st.slider("Number of components:", 2, len(selected_cols), 2))
            
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(data_scaled)
            
            # Explained variance
            st.subheader("📊 Explained Variance")
            col1, col2 = st.columns(2)
            
            with col1:
                explained_var = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Explained Variance Ratio': pca.explained_variance_ratio_,
                    'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
                })
                st.dataframe(explained_var)
            
            with col2:
                fig = px.bar(explained_var, x='Component', y='Explained Variance Ratio',
                           title='Explained Variance by Component')
                st.plotly_chart(fig, use_container_width=True)
            
            # PCA scatter plot
            if n_components >= 2:
                pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
                fig = px.scatter(pca_df, x='PC1', y='PC2', title='PCA Scatter Plot')
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature loadings
            st.subheader("🎯 Feature Loadings")
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=selected_cols
            )
            st.dataframe(loadings)
        
        elif analysis_type == "K-Means Clustering":
            st.subheader("🎯 K-Means Clustering")
            
            # Determine optimal number of clusters using elbow method
            max_k = min(10, len(df_clean) // 2)
            k_range = range(2, max_k + 1)
            inertias = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data_scaled)
                inertias.append(kmeans.inertia_)
            
            # Elbow plot
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(x=list(k_range), y=inertias, markers=True,
                            title='Elbow Method for Optimal k',
                            labels={'x': 'Number of Clusters', 'y': 'Inertia'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                optimal_k = st.slider("Select number of clusters:", 2, max_k, 3)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_scaled)
            
            # Add cluster labels to dataframe
            df_clustered = df_clean.copy()
            df_clustered['Cluster'] = clusters
            
            # Cluster visualization
            if len(selected_cols) >= 2:
                fig = px.scatter(df_clustered, x=selected_cols[0], y=selected_cols[1], 
                               color='Cluster', title='K-Means Clustering Results')
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster summary
            st.subheader("📊 Cluster Summary")
            cluster_summary = df_clustered.groupby('Cluster')[selected_cols].mean()
            st.dataframe(cluster_summary)
        
        elif analysis_type == "Hierarchical Clustering":
            st.subheader("🌳 Hierarchical Clustering")
            
            # Limit data size for performance
            if len(df_clean) > 1000:
                st.warning("Using sample of 1000 rows for performance.")
                sample_data = df_clean.sample(1000, random_state=42)
                sample_scaled = scaler.fit_transform(sample_data)
            else:
                sample_data = df_clean
                sample_scaled = data_scaled
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(sample_scaled, method='ward')
            
            # Dendrogram
            fig, ax = plt.subplots(figsize=(12, 8))
            dendrogram(linkage_matrix, ax=ax)
            ax.set_title('Hierarchical Clustering Dendrogram')
            st.pyplot(fig)
        
        elif analysis_type == "Outlier Detection":
            st.subheader("🚨 Outlier Detection")
            
            # Isolation Forest
            contamination = st.slider("Contamination rate:", 0.01, 0.5, 0.1)
            
            isolation_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = isolation_forest.fit_predict(data_scaled)
            
            # Add outlier labels
            df_outliers = df_clean.copy()
            df_outliers['Outlier'] = outliers == -1
            
            outlier_count = sum(outliers == -1)
            st.metric("Outliers Detected", f"{outlier_count} ({outlier_count/len(df_clean)*100:.2f}%)")
            
            # Visualization
            if len(selected_cols) >= 2:
                fig = px.scatter(df_outliers, x=selected_cols[0], y=selected_cols[1],
                               color='Outlier', title='Outlier Detection Results')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show outlier data
            if outlier_count > 0:
                st.subheader("🔍 Detected Outliers")
                outlier_data = df_outliers[df_outliers['Outlier']]
                st.dataframe(outlier_data[selected_cols])
    
    def data_cleaning(self):
        """Data cleaning and preprocessing tools"""
        st.markdown('<div class="section-header">🧹 Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            sheet_name = st.selectbox("Select sheet for cleaning:", list(self.data.keys()))
            df = self.data[sheet_name].copy()
        else:
            df = self.data.copy()
        
        # Missing data analysis
        st.subheader("❓ Missing Data Analysis")
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        }).sort_values('Missing Count', ascending=False)
        
        # Show only columns with missing data
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            # Missing data heatmap
            if len(missing_df) > 0:
                fig = px.bar(missing_df, x='Column', y='Missing Percentage',
                           title='Missing Data by Column')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Missing data handling options
            st.subheader("🔧 Handle Missing Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                missing_strategy = st.selectbox("Strategy for numeric columns:", [
                    "Drop rows with missing values",
                    "Fill with mean",
                    "Fill with median", 
                    "Fill with mode",
                    "Forward fill",
                    "Backward fill"
                ])
            
            with col2:
                categorical_strategy = st.selectbox("Strategy for categorical columns:", [
                    "Drop rows with missing values",
                    "Fill with mode",
                    "Fill with 'Unknown'",
                    "Forward fill",
                    "Backward fill"
                ])
            
            if st.button("Apply Missing Data Strategy"):
                df_cleaned = df.copy()
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                # Handle numeric columns
                for col in numeric_cols:
                    if missing_strategy == "Drop rows with missing values":
                        df_cleaned = df_cleaned.dropna(subset=[col])
                    elif missing_strategy == "Fill with mean":
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    elif missing_strategy == "Fill with median":
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    elif missing_strategy == "Fill with mode":
                        df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 0, inplace=True)
                    elif missing_strategy == "Forward fill":
                        df_cleaned[col].fillna(method='ffill', inplace=True)
                    elif missing_strategy == "Backward fill":
                        df_cleaned[col].fillna(method='bfill', inplace=True)
                
                # Handle categorical columns
                for col in categorical_cols:
                    if categorical_strategy == "Drop rows with missing values":
                        df_cleaned = df_cleaned.dropna(subset=[col])
                    elif categorical_strategy == "Fill with mode":
                        df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown', inplace=True)
                    elif categorical_strategy == "Fill with 'Unknown'":
                        df_cleaned[col].fillna('Unknown', inplace=True)
                    elif categorical_strategy == "Forward fill":
                        df_cleaned[col].fillna(method='ffill', inplace=True)
                    elif categorical_strategy == "Backward fill":
                        df_cleaned[col].fillna(method='bfill', inplace=True)
                
                st.success(f"Data cleaned! Rows: {len(df)} → {len(df_cleaned)}")
                
                # Update the data
                if isinstance(self.data, dict):
                    self.data[sheet_name] = df_cleaned
                else:
                    self.data = df_cleaned
                
                st.experimental_rerun()
        
        else:
            st.success("✅ No missing data found!")
        
        # Duplicate detection
        st.subheader("🔍 Duplicate Detection")
        
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)
        
        if duplicates > 0:
            if st.button("Remove Duplicates"):
                df_no_duplicates = df.drop_duplicates()
                st.success(f"Removed {duplicates} duplicate rows!")
                
                # Update the data
                if isinstance(self.data, dict):
                    self.data[sheet_name] = df_no_duplicates
                else:
                    self.data = df_no_duplicates
                
                st.experimental_rerun()
        
        # Data type conversion
        st.subheader("🔄 Data Type Conversion")
        
        column_to_convert = st.selectbox("Select column to convert:", df.columns)
        new_dtype = st.selectbox("New data type:", [
            "int64", "float64", "object", "category", "datetime64", "bool"
        ])
        
        if st.button("Convert Data Type"):
            try:
                if new_dtype == "datetime64":
                    df[column_to_convert] = pd.to_datetime(df[column_to_convert])
                elif new_dtype == "category":
                    df[column_to_convert] = df[column_to_convert].astype('category')
                else:
                    df[column_to_convert] = df[column_to_convert].astype(new_dtype)
                
                st.success(f"Converted {column_to_convert} to {new_dtype}")
                
                # Update the data
                if isinstance(self.data, dict):
                    self.data[sheet_name] = df
                else:
                    self.data = df
                
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")

def main():
    """Main application function"""
    st.markdown('<div class="main-header">🚀 Universal Data Analyzer Pro</div>', unsafe_allow_html=True)
    
    # Theme toggle
    col1, col2, col3 = st.columns([6, 1, 1])
    with col2:
        if st.button("🌓 Theme"):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.experimental_rerun()
    
    with col3:
        if st.button("🗑️ Clear Cache"):
            st.session_state.cached_data = {}
            st.success("Cache cleared!")
    
    # Initialize analyzer
    analyzer = UniversalDataAnalyzerPro()
    
    # Sidebar for file upload and navigation
    with st.sidebar:
        st.header("📁 Data Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls', 'json', 'xml', 'txt', 'parquet', 'zip'],
            help="Supported formats: CSV, Excel, JSON, XML, Text, Parquet, ZIP"
        )
        
        # Database connection option
        if st.checkbox("📊 Connect to Database"):
            analyzer.database_connector()
        
        # Navigation
        if uploaded_file is not None or analyzer.data is not None:
            st.header("🧭 Navigation")
            
            analysis_options = [
                "
            "📊 Data Overview",
                "📈 Statistical Analysis", 
                "🎨 Data Visualization",
                "🔬 Advanced Analytics",
                "🧹 Data Cleaning",
                "💾 Export Data",
                "📋 Generate Report"
            ]
            
            selected_analysis = st.radio("Select Analysis:", analysis_options)
    
    # Main content area
    if uploaded_file is not None:
        # Load and cache data
        if analyzer.load_data(uploaded_file):
            st.success("✅ Data loaded successfully!")
            
            # Display selected analysis
            if selected_analysis == "📊 Data Overview":
                analyzer.data_overview()
            elif selected_analysis == "📈 Statistical Analysis":
                analyzer.statistical_analysis()
            elif selected_analysis == "🎨 Data Visualization":
                analyzer.data_visualization()
            elif selected_analysis == "🔬 Advanced Analytics":
                analyzer.advanced_analysis()
            elif selected_analysis == "🧹 Data Cleaning":
                analyzer.data_cleaning()
            elif selected_analysis == "💾 Export Data":
                analyzer.export_data()
            elif selected_analysis == "📋 Generate Report":
                analyzer.generate_report()
        else:
            st.error("❌ Failed to load data. Please check your file format.")
    
    elif analyzer.data is not None:
        # Data loaded from database or other source
        if selected_analysis == "📊 Data Overview":
            analyzer.data_overview()
        elif selected_analysis == "📈 Statistical Analysis":
            analyzer.statistical_analysis()
        elif selected_analysis == "🎨 Data Visualization":
            analyzer.data_visualization()
        elif selected_analysis == "🔬 Advanced Analytics":
            analyzer.advanced_analysis()
        elif selected_analysis == "🧹 Data Cleaning":
            analyzer.data_cleaning()
        elif selected_analysis == "💾 Export Data":
            analyzer.export_data()
        elif selected_analysis == "📋 Generate Report":
            analyzer.generate_report()
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h1>🚀 Welcome to Universal Data Analyzer Pro</h1>
            <p style='font-size: 18px; color: #666;'>
                Your all-in-one solution for comprehensive data analysis
            </p>
            <br>
            <div style='display: flex; justify-content: center; gap: 40px; margin-top: 30px;'>
                <div style='text-align: center;'>
                    <h3>📁</h3>
                    <p>Upload Files</p>
                    <small>CSV, Excel, JSON, XML, Parquet</small>
                </div>
                <div style='text-align: center;'>
                    <h3>📊</h3>
                    <p>Analyze Data</p>
                    <small>Statistics, Visualization, ML</small>
                </div>
                <div style='text-align: center;'>
                    <h3>📋</h3>
                    <p>Generate Reports</p>
                    <small>Comprehensive Analysis Reports</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔍 Data Exploration
            - Comprehensive data overview
            - Missing data analysis
            - Data type detection
            - Sample data preview
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Statistical Analysis
            - Descriptive statistics
            - Distribution analysis
            - Correlation analysis
            - Normality tests
            """)
        
        with col3:
            st.markdown("""
            ### 🎨 Advanced Features
            - Interactive visualizations
            - Machine learning clustering
            - Principal Component Analysis
            - Outlier detection
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Universal Data Analyzer Pro | Built with Streamlit & Plotly"
        "</div>", 
        unsafe_allow_html=True
    )

def add_missing_methods_to_analyzer():
    """Add missing methods to the UniversalDataAnalyzerPro class"""
    
    def export_data(self):
        """Export processed data"""
        st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            sheet_name = st.selectbox("Select sheet to export:", list(self.data.keys()))
            df = self.data[sheet_name]
        else:
            df = self.data
        
        # Export format selection
        export_format = st.selectbox("Select export format:", ["CSV", "Excel", "JSON", "Parquet"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Preview data to be exported
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("📊 Export Summary")
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))
            st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Generate download
        if export_format == "CSV":
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="exported_data.csv",
                mime="text/csv"
            )
        
        elif export_format == "Excel":
            # Note: This requires openpyxl to be installed
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name="exported_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.error("Excel export requires 'openpyxl'. Please install it: pip install openpyxl")
        
        elif export_format == "JSON":
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="exported_data.json",
                mime="application/json"
            )
        
        elif export_format == "Parquet":
            # Note: This requires pyarrow to be installed
            try:
                parquet_buffer = io.BytesIO()
                df.to_parquet(parquet_buffer, index=False)
                parquet_data = parquet_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Parquet",
                    data=parquet_data,
                    file_name="exported_data.parquet",
                    mime="application/octet-stream"
                )
            except ImportError:
                st.error("Parquet export requires 'pyarrow'. Please install it: pip install pyarrow")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        st.markdown('<div class="section-header">📋 Analysis Report</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            sheet_name = st.selectbox("Select sheet for report:", list(self.data.keys()))
            df = self.data[sheet_name]
        else:
            df = self.data
        
        # Report sections
        report_sections = st.multiselect(
            "Select report sections:",
            ["Data Overview", "Statistical Summary", "Missing Data Analysis", 
             "Correlation Analysis", "Distribution Analysis", "Data Quality Assessment"],
            default=["Data Overview", "Statistical Summary", "Data Quality Assessment"]
        )
        
        if st.button("📄 Generate Report"):
            report_content = []
            
            # Header
            report_content.append("# 📊 Data Analysis Report\n")
            report_content.append(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_content.append(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n\n")
            
            if "Data Overview" in report_sections:
                report_content.append("## 📋 Data Overview\n")
                report_content.append(f"- **Total Rows:** {len(df):,}\n")
                report_content.append(f"- **Total Columns:** {len(df.columns)}\n")
                report_content.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
                report_content.append(f"- **Missing Data:** {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%\n\n")
                
                # Column info
                report_content.append("### Column Information\n")
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    unique_count = df[col].nunique()
                    report_content.append(f"- **{col}**: {dtype}, {null_count} nulls, {unique_count} unique values\n")
                report_content.append("\n")
            
            if "Statistical Summary" in report_sections:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    report_content.append("## 📈 Statistical Summary\n")
                    desc_stats = df[numeric_cols].describe()
                    report_content.append(desc_stats.to_markdown())
                    report_content.append("\n\n")
            
            if "Missing Data Analysis" in report_sections:
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    report_content.append("## ❓ Missing Data Analysis\n")
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Missing Percentage': (missing_data.values / len(df)) * 100
                    })
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    report_content.append(missing_df.to_markdown(index=False))
                    report_content.append("\n\n")
            
            if "Data Quality Assessment" in report_sections:
                report_content.append("## ✅ Data Quality Assessment\n")
                
                # Duplicates
                duplicates = df.duplicated().sum()
                report_content.append(f"- **Duplicate Rows:** {duplicates} ({duplicates/len(df)*100:.2f}%)\n")
                
                # Data completeness
                completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                report_content.append(f"- **Data Completeness:** {completeness:.2f}%\n")
                
                # Data consistency (basic checks)
                consistency_issues = []
                for col in df.select_dtypes(include=['object']).columns:
                    if df[col].str.contains(r'^\s+|\s+$', na=False).any():
                        consistency_issues.append(f"Leading/trailing spaces in '{col}'")
                
                if consistency_issues:
                    report_content.append("- **Consistency Issues:**\n")
                    for issue in consistency_issues:
                        report_content.append(f"  - {issue}\n")
                else:
                    report_content.append("- **Consistency Issues:** None detected\n")
                
                report_content.append("\n")
            
            # Combine report
            full_report = "".join(report_content)
            
            # Display report
            st.markdown("### 📄 Generated Report")
            st.markdown(full_report)
            
            # Download report
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=full_report,
                file_name="data_analysis_report.md",
                mime="text/markdown"
            )
    
    def database_connector(self):
        """Database connection interface"""
        st.subheader("🗄️ Database Connection")
        
        db_type = st.selectbox("Database Type:", ["SQLite", "PostgreSQL", "MySQL", "SQL Server"])
        
        if db_type == "SQLite":
            db_file = st.file_uploader("Upload SQLite file:", type=['db', 'sqlite', 'sqlite3'])
            if db_file:
                # Note: This would require sqlite3 and proper handling
                st.info("SQLite connection would be implemented here")
        
        else:
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host:", "localhost")
                port = st.number_input("Port:", value=5432 if db_type == "PostgreSQL" else 3306)
            with col2:
                database = st.text_input("Database:")
                username = st.text_input("Username:")
            
            password = st.text_input("Password:", type="password")
            
            if st.button("🔌 Connect"):
                st.info("Database connection would be implemented here")
                # Note: This would require appropriate database drivers (psycopg2, pymysql, etc.)

# Add methods to the class (this would normally be done within the class definition)
# For demonstration purposes, showing how the methods would be added
import io

if __name__ == "__main__":
    main()
