# universal_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="📊 Universal Analytics Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

# File upload
st.sidebar.header("📂 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not uploaded_file:
    st.warning("Please upload a dataset to begin.")
    st.stop()

df = load_data(uploaded_file)

# Clean column names
df.columns = df.columns.str.strip()

# Auto-detect types
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()
date_cols = df.select_dtypes(include='datetime').columns.tolist()

# Convert potential datetime columns
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            continue

# Sidebar filters
st.sidebar.header("🔍 Filters")

filters = {}
for col in cat_cols:
    options = df[col].dropna().unique().tolist()
    selected = st.sidebar.multiselect(f"{col}", options, default=options)
    filters[col] = selected

for col, values in filters.items():
    df = df[df[col].isin(values)]

# Main header
st.markdown("<h1 style='text-align:center;'>📊 Universal Analytics Dashboard</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📈 Visuals", "📊 Correlation"])

# === Tab 1: Overview ===
with tab1:
    st.header("📌 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    if num_cols:
        col2.metric("Total Numeric Columns", len(num_cols))
        col3.metric("Average of First Numeric", f"{df[num_cols[0]].mean():.2f}")
    st.dataframe(df.head(50), use_container_width=True)

# === Tab 2: Visuals ===
with tab2:
    st.header("📈 Interactive Visualizations")
    x_axis = st.selectbox("Select X-axis", df.columns)
    y_axis = st.selectbox("Select Y-axis (numeric)", num_cols)
    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Box", "Pie"])

    if chart_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis)
    elif chart_type == "Line":
        fig = px.line(df, x=x_axis, y=y_axis)
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=x_axis if x_axis in cat_cols else None)
    elif chart_type == "Box":
        fig = px.box(df, x=x_axis, y=y_axis)
    elif chart_type == "Pie":
        counts = df[x_axis].value_counts()
        fig = px.pie(names=counts.index, values=counts.values)

    st.plotly_chart(fig, use_container_width=True)

# === Tab 3: Correlation ===
with tab3:
    if len(num_cols) >= 2:
        st.header("📊 Correlation Heatmap")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("At least two numeric columns are needed for correlation analysis.")

# === Footer ===
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Built with ❤️ using Streamlit</div>",
    unsafe_allow_html=True
)
