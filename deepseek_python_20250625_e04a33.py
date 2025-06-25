# Improved Universal Data Analyzer Pro

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import zipfile
import sqlite3
import hashlib
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Enhanced imports with better error handling
class PackageManager:
    def __init__(self):
        self.available_packages = {
            'nltk': False,
            'wordcloud': False,
            'psycopg2': False,
            'mysql': False,
            'aggrid': False,
            'fpdf': False,
            'pdfkit': False,
            'statsmodels': False
        }
        self.check_packages()
    
    def check_packages(self):
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            self.available_packages['nltk'] = True
        except ImportError:
            pass
            
        try:
            from wordcloud import WordCloud
            self.available_packages['wordcloud'] = True
        except ImportError:
            pass
            
        try:
            import psycopg2
            self.available_packages['psycopg2'] = True
        except ImportError:
            pass
            
        try:
            import mysql.connector
            self.available_packages['mysql'] = True
        except ImportError:
            pass
            
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder
            self.available_packages['aggrid'] = True
        except ImportError:
            pass
            
        try:
            from fpdf import FPDF
            self.available_packages['fpdf'] = True
        except ImportError:
            pass
            
        try:
            import pdfkit
            self.available_packages['pdfkit'] = True
        except ImportError:
            pass
            
        try:
            import statsmodels.api as sm
            self.available_packages['statsmodels'] = True
        except ImportError:
            pass

# Initialize package manager
pm = PackageManager()

# Improved session state management
class AppState:
    def __init__(self):
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
        if 'cached_data' not in st.session_state:
            st.session_state.cached_data = {}
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "Data Overview"
    
    def set_theme(self, theme):
        st.session_state.theme = theme
    
    def get_theme(self):
        return st.session_state.theme
    
    def add_to_history(self, entry):
        st.session_state.analysis_history.append(entry)
    
    def get_history(self):
        return st.session_state.analysis_history
    
    def cache_data(self, key, value):
        st.session_state.cached_data[key] = value
    
    def get_cached_data(self, key):
        return st.session_state.cached_data.get(key)

# Enhanced theme management
class ThemeManager:
    @staticmethod
    def get_theme_css(theme):
        if theme == 'dark':
            return """
            <style>
                :root {
                    --primary: #00d4ff;
                    --secondary: #ff6b6b;
                    --bg-gradient: linear-gradient(135deg, #0c0c0c, #1a1a1a);
                    --card-bg: linear-gradient(135deg, #1e1e1e, #2d2d2d);
                    --text: #ffffff;
                    --border: #444;
                }
            </style>
            """
        else:
            return """
            <style>
                :root {
                    --primary: #1f77b4;
                    --secondary: #ff7f0e;
                    --bg-gradient: linear-gradient(135deg, #f0f2f6, #ffffff);
                    --card-bg: linear-gradient(135deg, #f0f2f6, #ffffff);
                    --text: #333333;
                    --border: #e0e0e0;
                }
            </style>
            """
    
    @staticmethod
    def apply_theme(theme):
        css = ThemeManager.get_theme_css(theme)
        st.markdown(css, unsafe_allow_html=True)
        st.markdown(f"""
        <style>
            .main-header {{
                font-size: 3rem;
                color: var(--primary);
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: {'0 0 10px var(--primary)' if theme == 'dark' else '2px 2px 4px rgba(0,0,0,0.1)'};
            }}
            .section-header {{
                font-size: 1.5rem;
                color: var(--secondary);
                margin-top: 2rem;
                margin-bottom: 1rem;
                border-left: 4px solid var(--secondary);
                padding-left: 1rem;
            }}
            .metric-container {{
                background: var(--card-bg);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border: 1px solid var(--border);
                box-shadow: {'0 4px 6px rgba(0, 0, 0, 0.3)' if theme == 'dark' else '0 2px 4px rgba(0,0,0,0.1)'};
            }}
            .stApp {{
                background: var(--bg-gradient);
            }}
            .nlp-insight {{
                background: linear-gradient(45deg, #2d1b69, #11998e);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
            }}
        </style>
        """, unsafe_allow_html=True)

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Data Analyzer Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize app state and theme
app_state = AppState()
ThemeManager.apply_theme(app_state.get_theme())

class EnhancedDataAnalyzer:
    def __init__(self):
        self.data = None
        self.file_info = {
            'name': None,
            'type': None,
            'hash': None,
            'sheets': None
        }
        self.analysis_results = {}

    def _detect_file_type(self, file):
        """Detect file type based on extension"""
        filename = file.name.lower()
        if filename.endswith(('.csv', '.tsv')):
            return 'csv'
        elif filename.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif filename.endswith('.json'):
            return 'json'
        elif filename.endswith('.xml'):
            return 'xml'
        elif filename.endswith('.txt'):
            return 'text'
        elif filename.endswith('.parquet'):
            return 'parquet'
        elif filename.endswith('.zip'):
            return 'zip'
        else:
            return 'unknown'

    def _load_csv(self, file):
        """Load CSV file with multiple encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        delimiters = [',', ';', '\t', '|']
        
        for enc in encodings:
            for delim in delimiters:
                try:
                    file.seek(0)
                    self.data = pd.read_csv(file, encoding=enc, delimiter=delim)
                    if len(self.data.columns) > 1:
                        self.data = self._enhance_data(self.data)
                        return True
                except:
                    continue
        return False

    def _load_excel(self, file):
        """Load Excel file with support for multiple sheets"""
        try:
            excel_file = pd.ExcelFile(file)
            sheets = excel_file.sheet_names
            if len(sheets) == 1:
                self.data = self._enhance_data(pd.read_excel(file))
            else:
                self.data = {sheet: self._enhance_data(pd.read_excel(file, sheet_name=sheet)) 
                           for sheet in sheets}
                self.file_info['sheets'] = sheets
            return True
        except Exception as e:
            st.error(f"Excel loading error: {str(e)}")
            return False

    def _load_json(self, file):
        """Load JSON file"""
        try:
            file.seek(0)
            json_data = json.load(file)
            if isinstance(json_data, list):
                self.data = pd.json_normalize(json_data)
            elif isinstance(json_data, dict):
                self.data = pd.json_normalize([json_data])
            else:
                self.data = pd.DataFrame({'data': [json_data]})
            self.data = self._enhance_data(self.data)
            return True
        except Exception as e:
            st.error(f"JSON loading error: {str(e)}")
            return False

    def _load_xml(self, file):
        """Load XML file"""
        try:
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
            self.data = self._enhance_data(self.data)
            return True
        except Exception as e:
            st.error(f"XML loading error: {str(e)}")
            return False

    def _load_parquet(self, file):
        """Load Parquet file"""
        try:
            file.seek(0)
            self.data = pd.read_parquet(file)
            self.data = self._enhance_data(self.data)
            return True
        except Exception as e:
            st.error(f"Parquet loading error: {str(e)}")
            return False

    def _load_text(self, file):
        """Load text file"""
        try:
            file.seek(0)
            content = file.read().decode('utf-8')
            lines = content.split('\n')
            self.data = pd.DataFrame({'text': lines})
            return True
        except Exception as e:
            st.error(f"Text file loading error: {str(e)}")
            return False

    def _load_zip(self, file):
        """Load data from ZIP archive"""
        try:
            file.seek(0)
            with zipfile.ZipFile(file, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                st.info(f"ZIP file contains: {', '.join(file_list)}")
                for filename in file_list:
                    if filename.endswith(('.csv', '.xlsx', '.json')):
                        with zip_ref.open(filename) as extracted_file:
                            if filename.endswith('.csv'):
                                return self._load_csv(extracted_file)
                            elif filename.endswith('.xlsx'):
                                return self._load_excel(extracted_file)
                            elif filename.endswith('.json'):
                                return self._load_json(extracted_file)
                return False
        except Exception as e:
            st.error(f"ZIP file loading error: {str(e)}")
            return False

    def _load_unknown(self, file):
        """Handler for unknown file types"""
        st.error(f"Unsupported file type: {self.file_info['type']}")
        return False

    def _enhance_data(self, df):
        """Perform automatic data enhancements"""
        # Date detection
        date_cols = [col for col in df.columns 
                    if any(keyword in col.lower() 
                          for keyword in ['date', 'time', 'created', 'modified'])]
        
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Numeric conversion
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        return df

    def _cache_data(self):
        """Cache the loaded data"""
        cache_entry = {
            'data': self.data,
            'file_info': self.file_info,
            'timestamp': datetime.now()
        }
        app_state.cache_data(self.file_info['hash'], cache_entry)

    def load_data(self, file):
        """Main data loading method with caching"""
        if file is None:
            return False
            
        self.file_info = {
            'name': file.name,
            'type': self._detect_file_type(file),
            'hash': hashlib.md5(file.read()).hexdigest(),
            'sheets': None
        }
        file.seek(0)  # Reset file pointer after reading for hash
        
        # Check cache first
        cached = app_state.get_cached_data(self.file_info['hash'])
        if cached:
            st.info("📦 Loading from cache...")
            self.data = cached['data']
            self.file_info.update(cached['file_info'])
            return True
        
        try:
            loader = {
                'csv': self._load_csv,
                'excel': self._load_excel,
                'json': self._load_json,
                'xml': self._load_xml,
                'parquet': self._load_parquet,
                'text': self._load_text,
                'zip': self._load_zip
            }.get(self.file_info['type'], self._load_unknown)
            
            success = loader(file)
            if success:
                self._cache_data()
            return success
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False    # [Other loader methods (_load_json, _load_xml, etc.) would follow similar patterns]
    
    def _enhance_data(self, df):
        # Enhanced date detection and type conversion
        df = self._detect_dates(df)
        # Automatic type conversion for numeric columns stored as strings
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        return df
    
    # [Rest of the EnhancedDataAnalyzer class implementation...]

# Main application class with improved UI/UX
class DataAnalyzerApp:
    def __init__(self):
        self.analyzer = EnhancedDataAnalyzer()
        self.setup_sidebar()
        self.setup_main()
    
    def setup_sidebar(self):
        with st.sidebar:
            st.title("🔧 Settings")
            
            # Theme toggle
            current_theme = app_state.get_theme()
            if st.button(f"Switch to {'Dark' if current_theme == 'light' else 'Light'} Theme"):
                new_theme = 'dark' if current_theme == 'light' else 'light'
                app_state.set_theme(new_theme)
                ThemeManager.apply_theme(new_theme)
                st.experimental_rerun()
            
            # File upload
            st.markdown("## 📂 Data Input")
            uploaded_file = st.file_uploader(
                "Upload your data file", 
                type=['csv', 'xlsx', 'json', 'xml', 'txt', 'parquet', 'zip'],
                key="file_uploader"
            )
            
            if uploaded_file:
                if self.analyzer.load_data(uploaded_file):
                    st.success(f"Successfully loaded {uploaded_file.name}")
                    app_state.current_tab = "Data Overview"
            
            # Database connection option
            if st.checkbox("Connect to Database"):
                self.show_database_connector()
            
            # About section
            st.markdown("---")
            st.markdown("""
            ### About Enhanced Data Analyzer
            **Features:**
            - Advanced data loading
            - Smart data cleaning
            - Interactive visualizations
            - Machine learning integration
            - Comprehensive reporting
            
            Created with ❤️ using Streamlit
            """)
    
    def setup_main(self):
        st.markdown('<div class="main-header">Enhanced Data Analyzer Pro</div>', unsafe_allow_html=True)
        
        if not hasattr(self.analyzer, 'data') or self.analyzer.data is None:
            st.info("Please upload a data file to begin analysis")
            return
        
        tabs = [
            "Data Overview", 
            "Data Cleaning", 
            "Exploratory Analysis", 
            "Advanced Analytics",
            "NLP Analysis",
            "Reports"
        ]
        
        current_tab = st.radio(
            "Select Analysis Mode:",
            tabs,
            horizontal=True,
            index=tabs.index(app_state.current_tab)
        )
        
        app_state.current_tab = current_tab
        
        if current_tab == "Data Overview":
            self.show_data_overview()
        elif current_tab == "Data Cleaning":
            self.show_data_cleaning()
        elif current_tab == "Exploratory Analysis":
            self.show_exploratory_analysis()
        elif current_tab == "Advanced Analytics":
            self.show_advanced_analysis()
        elif current_tab == "NLP Analysis":
            self.show_nlp_analysis()
        elif current_tab == "Reports":
            self.show_report_generator()
    
    def show_data_overview(self):
        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
        
        if isinstance(self.analyzer.data, dict):
            sheet = st.selectbox("Select sheet:", list(self.analyzer.data.keys()))
            df = self.analyzer.data[sheet]
        else:
            df = self.analyzer.data
        
        # Improved overview with expandable sections
        with st.expander("Basic Information", expanded=True):
            cols = st.columns(4)
            cols[0].metric("Rows", f"{len(df):,}")
            cols[1].metric("Columns", len(df.columns))
            cols[2].metric("Missing Values", f"{df.isnull().sum().sum():,}")
            cols[3].metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with st.expander("Data Preview"):
            st.dataframe(df.head(), use_container_width=True)
        
        with st.expander("Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Missing %': (df.isnull().mean() * 100).round(2),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        with st.expander("Quick Statistics"):
            tab1, tab2 = st.tabs(["Numeric", "Categorical"])
            with tab1:
                num_cols = df.select_dtypes(include=np.number).columns
                if len(num_cols) > 0:
                    st.dataframe(df[num_cols].describe().T, use_container_width=True)
                else:
                    st.info("No numeric columns found")
            with tab2:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    stats = []
                    for col in cat_cols:
                        stats.append({
                            'Column': col,
                            'Unique Values': df[col].nunique(),
                            'Most Common': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                            'Frequency': df[col].value_counts().iloc[0] if not df[col].mode().empty else 0
                        })
                    st.dataframe(pd.DataFrame(stats), use_container_width=True)
                else:
                    st.info("No categorical columns found")
    
    # [Other methods for different tabs would follow...]
    
    def show_exploratory_analysis(self):
        st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        if isinstance(self.analyzer.data, dict):
            sheet = st.selectbox("Select sheet:", list(self.analyzer.data.keys()), key="eda_sheet")
            df = self.analyzer.data[sheet]
        else:
            df = self.analyzer.data
        
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["Distribution Analysis", "Correlation Analysis", "Categorical Analysis"],
            key="eda_type"
        )
        
        if analysis_type == "Distribution Analysis":
            self.show_distribution_analysis(df)
        elif analysis_type == "Correlation Analysis":
            self.show_correlation_analysis(df)
        elif analysis_type == "Categorical Analysis":
            self.show_categorical_analysis(df)
    
    def show_distribution_analysis(self, df):
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            st.warning("No numeric columns found for distribution analysis")
            return
        
        col = st.selectbox("Select numeric column:", num_cols, key="dist_col")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=col, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)
                # Enhanced normality check with Q-Q plot
        st.subheader("Normality Check")
        qq_data = df[col].dropna()
        qq_data = (qq_data - qq_data.mean()) / qq_data.std()
        
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
            title=f"Q-Q Plot for {col}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_correlation_analysis(self, df):
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis")
            return
        
        method = st.selectbox(
            "Correlation method:", 
            ['pearson', 'spearman', 'kendall'],
            key="corr_method"
        )
        
        # Enhanced correlation matrix with clustering
        corr = df[num_cols].corr(method=method)
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title=f"Correlation Matrix ({method.capitalize()})",
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Find and visualize high correlations
        threshold = st.slider(
            "Highlight correlations above:", 
            0.5, 0.95, 0.7, 0.05,
            key="corr_threshold"
        )
        
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > threshold:
                    high_corr.append({
                        'Variable 1': corr.columns[i],
                        'Variable 2': corr.columns[j],
                        'Correlation': corr.iloc[i, j]
                    })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr)
            st.dataframe(
                high_corr_df.sort_values('Correlation', ascending=False), 
                use_container_width=True
            )
            
            # Visualize strongest correlation
            strongest = high_corr_df.iloc[0]
            try:
                fig = px.scatter(
                    df,
                    x=strongest['Variable 1'],
                    y=strongest['Variable 2'],
                    title=f"Strongest Correlation: {strongest['Variable 1']} vs {strongest['Variable 2']}",
                    trendline="ols" if pm.available_packages['statsmodels'] else None
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if not pm.available_packages['statsmodels']:
                    st.warning(
                        "Trendline requires statsmodels. Install with: pip install statsmodels"
                    )
            except Exception as e:
                st.error(f"Error creating scatter plot: {str(e)}")
        else:
            st.info(f"No correlations above {threshold} found")
    
    # [Other methods would continue...]

# Run the app
if __name__ == "__main__":
    app = DataAnalyzerApp()
