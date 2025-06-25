import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# Set page config
st.set_page_config(page_title="Universal Dashboard", layout="wide")

# Title
st.title("📊 Universal Streamlit Dashboard")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Load data
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Data successfully loaded!")
    
    # Preview
    st.subheader("📁 Data Preview")
    st.dataframe(df.head())

    # Filters
    st.sidebar.subheader("🔍 Filters")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    category_cols = df.select_dtypes(include='object').columns.tolist()

    if category_cols:
        selected_category = st.sidebar.selectbox("Select category column to filter", category_cols)
        selected_values = st.sidebar.multiselect("Choose values", df[selected_category].unique())
        if selected_values:
            df = df[df[selected_category].isin(selected_values)]

    # KPIs
    st.subheader("📌 KPI Metrics")
    if numeric_cols:
        col1, col2, col3 = st.columns(3)
        col1.metric("Row Count", len(df))
        col2.metric(f"{numeric_cols[0]} Mean", round(df[numeric_cols[0]].mean(), 2))
        col3.metric(f"{numeric_cols[0]} Max", df[numeric_cols[0]].max())

    # Charts
    st.subheader("📈 Visualizations")
    colx1, colx2 = st.columns(2)

    with colx1:
        if category_cols and numeric_cols:
            st.markdown("### Bar Chart")
            fig1 = px.bar(df, x=category_cols[0], y=numeric_cols[0], color=category_cols[0])
            st.plotly_chart(fig1, use_container_width=True)

    with colx2:
        if len(numeric_cols) >= 2:
            st.markdown("### Scatter Plot")
            fig2 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=category_cols[0] if category_cols else None)
            st.plotly_chart(fig2, use_container_width=True)

    # Download filtered data
    st.subheader("⬇️ Download Filtered Data")
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download as CSV",
        data=convert_df(df),
        file_name='filtered_data.csv',
        mime='text/csv'
    )

else:
    st.warning("Upload a file to get started.")
