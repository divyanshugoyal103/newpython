import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
from st_aggrid import AgGrid, GridOptionsBuilder
import base64

# Page config
st.set_page_config(page_title="Advanced Dashboard", layout="wide")

st.title("📊 Advanced Universal Streamlit Dashboard")

# Upload file
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV or Excel", type=["csv", "xlsx"])

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Main logic
if uploaded_file:
    df = load_data(uploaded_file)

    st.success("✅ Data Loaded Successfully!")
    st.markdown("### 🧾 Raw Data Preview")
    AgGrid(df.head(100))

    # Column categorization
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Sidebar Filters
    st.sidebar.header("🔍 Filter Data")
    for col in cat_cols:
        options = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"{col}", options, default=options)
        df = df[df[col].isin(selected)]

    # KPIs
    st.markdown("### 📌 KPI Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    if numeric_cols:
        col2.metric(f"{numeric_cols[0]} Mean", round(df[numeric_cols[0]].mean(), 2))
        col3.metric(f"{numeric_cols[0]} Std Dev", round(df[numeric_cols[0]].std(), 2))

    # Visuals
    st.markdown("### 📈 Visualizations")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        chart_type = st.selectbox("Choose Chart Type", ["Bar", "Line", "Box", "Scatter", "Pie"])
        x_col = st.selectbox("X-axis", options=cat_cols + numeric_cols)
        y_col = st.selectbox("Y-axis", options=numeric_cols)

        if chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col, color=x_col)
        elif chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col)
        elif chart_type == "Box":
            fig = px.box(df, x=x_col, y=y_col)
        elif chart_type == "Scatter":
            color_col = st.selectbox("Color By", options=[None] + cat_cols)
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Pie":
            fig = px.pie(df, names=x_col, values=y_col)

        st.plotly_chart(fig, use_container_width=True)

    with plot_col2:
        if len(numeric_cols) >= 2:
            st.markdown("### 🔗 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # Pivot Table
    st.markdown("### 🧮 Pivot Table")
    pivot_index = st.multiselect("Pivot Index (Rows)", cat_cols)
    pivot_columns = st.multiselect("Pivot Columns", cat_cols)
    pivot_values = st.multiselect("Values", numeric_cols)
    aggfunc = st.selectbox("Aggregation Function", ["mean", "sum", "count", "min", "max"])

    if pivot_index and pivot_values:
        try:
            pivot_table = pd.pivot_table(
                df,
                index=pivot_index,
                columns=pivot_columns if pivot_columns else None,
                values=pivot_values,
                aggfunc=aggfunc,
                fill_value=0
            )
            st.dataframe(pivot_table)
        except Exception as e:
            st.error(f"Pivot Table Error: {e}")

    # Data export
    st.markdown("### ⬇️ Download Processed Data")

    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.save()
        return output.getvalue()

    csv = df.to_csv(index=False).encode()
    excel = to_excel(df)

    colx, coly = st.columns(2)
    colx.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
    coly.download_button("Download Excel", excel, "filtered_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Upload a file from the sidebar to begin 🚀")
