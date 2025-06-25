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
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Universal Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class UniversalDataAnalyzer:
    def __init__(self):
        self.data = None
        self.file_type = None
        self.file_name = None
        
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
    
    def load_data(self, file):
        """Load data based on file type"""
        self.file_name = file.name
        self.file_type = self.detect_file_type(file)
        
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
                                                  low_memory=False, parse_dates=True, 
                                                  infer_datetime_format=True)
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
                    self.data = pd.read_excel(file, parse_dates=True)
                else:
                    # Handle multiple sheets
                    sheet_data = {}
                    for sheet in excel_file.sheet_names:
                        sheet_data[sheet] = pd.read_excel(file, sheet_name=sheet, parse_dates=True)
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
                                
            return True
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
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
        # Data types
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique(),
            'Sample Values': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'NaN' for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Preview data
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    def statistical_analysis(self):
        """Perform statistical analysis"""
        st.markdown('<div class="section-header">📈 Statistical Analysis</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            for sheet_name, df in self.data.items():
                st.subheader(f"Analysis for Sheet: {sheet_name}")
                self._perform_statistical_analysis(df)
        else:
            self._perform_statistical_analysis(self.data)
    
    def _perform_statistical_analysis(self, df):
        """Helper function for statistical analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            st.subheader("Descriptive Statistics - Numeric Columns")
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
            
            # Correlation matrix
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                               color_continuous_scale='RdBu',
                               aspect='auto',
                               title='Correlation Heatmap')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        if len(categorical_cols) > 0:
            st.subheader("Categorical Columns Summary")
            cat_summary = pd.DataFrame({
                'Column': categorical_cols,
                'Unique Values': [df[col].nunique() for col in categorical_cols],
                'Most Common': [df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A' for col in categorical_cols],
                'Most Common Count': [df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0 for col in categorical_cols]
            })
            st.dataframe(cat_summary, use_container_width=True)
    
    def visualizations(self):
        """Create comprehensive visualizations"""
        st.markdown('<div class="section-header">📊 Advanced Visualizations</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            for sheet_name, df in self.data.items():
                st.subheader(f"Visualizations for Sheet: {sheet_name}")
                self._create_visualizations(df)
        else:
            self._create_visualizations(self.data)
    
    def _create_visualizations(self, df):
        """Helper function to create visualizations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        # Distribution plots for numeric columns
        if len(numeric_cols) > 0:
            st.subheader("Distribution Analysis")
            
            # Select columns for visualization
            selected_numeric = st.multiselect(
                "Select numeric columns for distribution analysis:",
                numeric_cols.tolist(),
                default=numeric_cols.tolist()[:3]  # Default to first 3
            )
            
            if selected_numeric:
                # Histograms
                fig = make_subplots(
                    rows=(len(selected_numeric) + 2) // 3,
                    cols=3,
                    subplot_titles=selected_numeric
                )
                
                for i, col in enumerate(selected_numeric):
                    row = (i // 3) + 1
                    col_idx = (i % 3) + 1
                    
                    fig.add_trace(
                        go.Histogram(x=df[col], name=col, showlegend=False),
                        row=row, col=col_idx
                    )
                
                fig.update_layout(height=300 * ((len(selected_numeric) + 2) // 3))
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plots
                if len(selected_numeric) > 1:
                    st.subheader("Box Plots")
                    fig = go.Figure()
                    for col in selected_numeric:
                        fig.add_trace(go.Box(y=df[col], name=col))
                    fig.update_layout(title="Box Plots for Numeric Columns")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Categorical analysis
        if len(categorical_cols) > 0:
            st.subheader("Categorical Analysis")
            
            selected_cat = st.selectbox(
                "Select categorical column for analysis:",
                categorical_cols.tolist()
            )
            
            if selected_cat:
                value_counts = df[selected_cat].value_counts().head(20)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top Categories in {selected_cat}"
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {selected_cat}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            st.subheader("Time Series Analysis")
            
            date_col = st.selectbox("Select date column:", date_cols.tolist())
            value_col = st.selectbox("Select value column:", numeric_cols.tolist())
            
            if date_col and value_col:
                # Sort by date
                df_sorted = df.sort_values(date_col)
                
                fig = px.line(
                    df_sorted,
                    x=date_col,
                    y=value_col,
                    title=f"{value_col} over time"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot matrix for numeric data
        if len(numeric_cols) >= 2:
            st.subheader("Relationships Between Variables")
            
            if len(numeric_cols) <= 5:
                # Scatter plot matrix
                fig = px.scatter_matrix(
                    df[numeric_cols],
                    title="Scatter Plot Matrix"
                )
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Allow user to select columns
                x_col = st.selectbox("Select X-axis:", numeric_cols.tolist())
                y_col = st.selectbox("Select Y-axis:", numeric_cols.tolist())
                
                if x_col != y_col:
                    color_col = None
                    if len(categorical_cols) > 0:
                        color_col = st.selectbox("Color by (optional):", ['None'] + categorical_cols.tolist())
                        if color_col == 'None':
                            color_col = None
                    
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f"{y_col} vs {x_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def advanced_analysis(self):
        """Perform advanced analysis including ML techniques"""
        st.markdown('<div class="section-header">🧠 Advanced Analysis</div>', unsafe_allow_html=True)
        
        if isinstance(self.data, dict):
            for sheet_name, df in self.data.items():
                st.subheader(f"Advanced Analysis for Sheet: {sheet_name}")
                self._perform_advanced_analysis(df)
        else:
            self._perform_advanced_analysis(self.data)
    
    def _perform_advanced_analysis(self, df):
        """Helper function for advanced analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.warning("Advanced analysis requires at least 2 numeric columns.")
            return
        
        # Prepare data for ML analysis
        df_numeric = df[numeric_cols].dropna()
        
        if len(df_numeric) == 0:
            st.warning("No complete numeric data available for advanced analysis.")
            return
        
        # PCA Analysis
        st.subheader("Principal Component Analysis (PCA)")
        
        if len(numeric_cols) >= 2:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df_numeric)
            
            pca = PCA()
            pca_result = pca.fit_transform(data_scaled)
            
            # Explained variance
            fig = px.bar(
                x=range(1, len(pca.explained_variance_ratio_) + 1),
                y=pca.explained_variance_ratio_,
                title="PCA - Explained Variance Ratio",
                labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA scatter plot (first two components)
            if len(pca_result) > 0:
                pca_df = pd.DataFrame({
                    'PC1': pca_result[:, 0],
                    'PC2': pca_result[:, 1] if pca_result.shape[1] > 1 else [0] * len(pca_result)
                })
                
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    title="PCA - First Two Components"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Clustering Analysis
        st.subheader("Clustering Analysis")
        
        if len(df_numeric) >= 10:  # Need sufficient data for clustering
            # K-means clustering
            max_clusters = min(10, len(df_numeric) // 2)
            n_clusters = st.slider("Number of clusters:", 2, max_clusters, 3)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(data_scaled)
            
            # Add clusters to PCA plot
            if 'pca_df' in locals():
                pca_df['Cluster'] = clusters.astype(str)
                
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title=f"K-means Clustering (k={n_clusters})"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster centers
            centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
            centers_df = pd.DataFrame(centers_original, columns=numeric_cols)
            centers_df.index = [f'Cluster {i}' for i in range(n_clusters)]
            
            st.subheader("Cluster Centers")
            st.dataframe(centers_df.round(3), use_container_width=True)
        
        # Outlier Detection
        st.subheader("Outlier Detection")
        
        if len(df_numeric) >= 20:  # Need sufficient data for outlier detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data_scaled)
            
            outlier_count = sum(outliers == -1)
            st.metric("Detected Outliers", f"{outlier_count} ({outlier_count/len(df_numeric)*100:.1f}%)")
            
            # Show outliers in PCA space if available
            if 'pca_df' in locals():
                pca_df['Outlier'] = ['Outlier' if x == -1 else 'Normal' for x in outliers]
                
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Outlier',
                    title="Outlier Detection in PCA Space"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Statistical Tests
        st.subheader("Statistical Tests")
        
        if len(numeric_cols) >= 2:
            # Normality tests
            st.write("**Normality Tests (Shapiro-Wilk):**")
            normality_results = []
            
            for col in numeric_cols[:5]:  # Limit to first 5 columns
                if len(df_numeric[col]) <= 5000:  # Shapiro-Wilk works best with smaller samples
                    stat, p_value = stats.shapiro(df_numeric[col].sample(min(1000, len(df_numeric))))
                    normality_results.append({
                        'Column': col,
                        'Statistic': stat,
                        'P-value': p_value,
                        'Normal?': 'Yes' if p_value > 0.05 else 'No'
                    })
            
            if normality_results:
                st.dataframe(pd.DataFrame(normality_results), use_container_width=True)

def main():
    # Header
    st.markdown('<div class="main-header">🚀 Universal Data Analyzer</div>', unsafe_allow_html=True)
    st.markdown("*Upload any data file and get comprehensive analysis automatically*")
    
    # Initialize analyzer
    analyzer = UniversalDataAnalyzer()
    
    # Sidebar
    st.sidebar.header("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'xml', 'txt', 'parquet', 'zip', 'tsv'],
        help="Supports CSV, Excel, JSON, XML, TXT, Parquet, and ZIP files"
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading and analyzing your data..."):
            if analyzer.load_data(uploaded_file):
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
                
                # Analysis options
                st.sidebar.header("📊 Analysis Options")
                show_basic = st.sidebar.checkbox("Basic Information", value=True)
                show_stats = st.sidebar.checkbox("Statistical Analysis", value=True)
                show_viz = st.sidebar.checkbox("Visualizations", value=True)
                show_advanced = st.sidebar.checkbox("Advanced Analysis", value=True)
                
                # Perform analyses
                if show_basic:
                    analyzer.basic_info()
                
                if show_stats:
                    analyzer.statistical_analysis()
                
                if show_viz:
                    analyzer.visualizations()
                
                if show_advanced:
                    analyzer.advanced_analysis()
                
                # Download processed data option
                if hasattr(analyzer, 'data') and analyzer.data is not None:
                    if not isinstance(analyzer.data, dict):
                        st.sidebar.header("💾 Download")
                        csv_data = analyzer.data.to_csv(index=False)
                        st.sidebar.download_button(
                            label="Download Processed Data as CSV",
                            data=csv_data,
                            file_name=f"processed_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv"
                        )
            else:
                st.error("❌ Failed to load the file. Please check the file format and try again.")
    
    else:
        # Instructions
        st.info("👆 Upload a file using the sidebar to get started!")
        
        st.markdown("""
        ## 🎯 What This Tool Can Do:
        
        **Supported File Types:**
        - CSV/TSV files with automatic delimiter detection
        - Excel files (single or multiple sheets)
        - JSON files (nested structures supported)
        - XML files with automatic parsing
        - Parquet files
        - ZIP archives containing data files
        - Plain text files
        
        **Automatic Analysis Includes:**
        
        **📋 Basic Overview:**
        - Dataset dimensions and memory usage
        - Column types and missing data analysis
        - Data preview and sample values
        
        **📈 Statistical Analysis:**
        - Descriptive statistics for numeric columns
        - Correlation analysis with heatmaps
        - Categorical data summaries
        
        **📊 Visualizations:**
        - Distribution plots (histograms, box plots)
        - Categorical analysis (bar charts, pie charts)
        - Time series analysis (if date columns detected)
        - Scatter plots and relationship analysis
        
        **🧠 Advanced Analysis:**
        - Principal Component Analysis (PCA)
        - K-means clustering with interactive parameters
        - Outlier detection using Isolation Forest
        - Statistical normality tests
        
        **✨ Key Features:**
        - Automatic file type detection and parsing
        - Handles missing data gracefully
        - Interactive visualizations with Plotly
        - Multiple encoding support for text files
        - Multi-sheet Excel file support
        - ZIP file extraction and analysis
        """)

if __name__ == "__main__":
    main()
