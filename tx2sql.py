import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Cleaning App", layout="wide")

st.title("🧹 Data Cleaning App")

# 1. File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Raw Data")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Data Cleaning Options")

    # Drop or fill NA
    na_option = st.selectbox("Handle Missing Values", ["Do nothing", "Drop rows", "Fill with mean", "Fill with zero"])
    if na_option == "Drop rows":
        df.dropna(inplace=True)
    elif na_option == "Fill with mean":
        df.fillna(df.mean(numeric_only=True), inplace=True)
    elif na_option == "Fill with zero":
        df.fillna(0, inplace=True)

    # Remove duplicates
    if st.checkbox("Remove Duplicates"):
        df.drop_duplicates(inplace=True)

    # Rename columns
    st.subheader("Rename Columns")
    new_col_names = {}
    for col in df.columns:
        new_name = st.text_input(f"Rename column '{col}'", value=col)
        new_col_names[col] = new_name
    df.rename(columns=new_col_names, inplace=True)

    # Change data types
    st.subheader("Change Column Data Types")
    for col in df.select_dtypes(include=["object", "int64", "float64"]).columns:
        dtype = st.selectbox(f"Change data type of '{col}'", ["No change", "string", "int", "float"])
        try:
            if dtype == "string":
                df[col] = df[col].astype(str)
            elif dtype == "int":
                df[col] = df[col].astype(int)
            elif dtype == "float":
                df[col] = df[col].astype(float)
        except Exception as e:
            st.warning(f"Could not convert column {col}: {e}")

    # String cleaning
    if st.checkbox("Trim whitespace and lowercase strings"):
        df = df.apply(lambda x: x.str.strip().str.lower() if x.dtypes == "object" else x)

    st.subheader("🔍 Cleaned Data Preview")
    st.dataframe(df.head())

    # Download cleaned CSV
    st.download_button("Download Cleaned CSV", df.to_csv(index=False), "cleaned_data.csv", "text/csv")

else:
    st.info("Please upload a CSV file to begin.")
