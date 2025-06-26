import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(layout="wide")
st.title("📊 Descriptive Data Analysis Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Try reading based on file extension
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # Select numerical and categorical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Plot selector
        st.subheader("📈 Create a Plot")
        plot_type = st.selectbox("Choose Plot Type", ["Line", "Bar", "Histogram", "Box", "Scatter"])

        if plot_type == "Line":
            x_axis = st.selectbox("X-axis", df.columns)
            y_axis = st.selectbox("Y-axis", numeric_cols)
            fig = px.line(df, x=x_axis, y=y_axis)
        elif plot_type == "Bar":
            x_axis = st.selectbox("X-axis", cat_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols)
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif plot_type == "Histogram":
            col = st.selectbox("Column", numeric_cols)
            fig = px.histogram(df, x=col)
        elif plot_type == "Box":
            x_axis = st.selectbox("X-axis", cat_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols)
            fig = px.box(df, x=x_axis, y=y_axis)
        elif plot_type == "Scatter":
            x_axis = st.selectbox("X-axis", numeric_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
            fig = px.scatter(df, x=x_axis, y=y_axis, color=cat_cols[0] if cat_cols else None)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("Upload a CSV or Excel file to begin analysis.")

