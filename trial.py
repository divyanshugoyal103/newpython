import streamlit as st
import pandas as pd
import plotly.express as px

# Cache the data loading function for efficiency
@st.cache_data
def load_data(uploaded_file, sample_frac=1.0):
    # Sample fraction controls whether you load full or sample (1.0 = full data)
    if uploaded_file.name.endswith('.csv'):
        if sample_frac < 1.0:
            # For large CSVs, read in chunks and sample rows from chunks
            chunks = pd.read_csv(uploaded_file, chunksize=100000)
            sampled_chunks = [chunk.sample(frac=sample_frac) for chunk in chunks]
            df = pd.concat(sampled_chunks)
        else:
            df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac)
    return df

def show_data_overview(df):
    st.subheader("Data Overview")
    st.write("Shape:", df.shape)
    st.write("Columns and types:")
    st.write(df.dtypes)
    st.write("Summary Statistics:")
    st.write(df.describe(include='all'))

def plot_basic_histogram(df):
    st.subheader("Sample Histogram")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        col = st.selectbox("Select column to plot histogram", numeric_cols)
        fig = px.histogram(df, x=col, nbins=50, title=f'Histogram of {col}')
        st.plotly_chart(fig)
    else:
        st.write("No numeric columns to plot.")

def main():
    st.title("Deep Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Let user decide if they want to sample (to reduce memory/time)
        sample_frac = st.slider("Sample fraction of data to load (for faster analysis)", 
                                min_value=0.01, max_value=1.0, value=1.0, step=0.01)

        try:
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file, sample_frac=sample_frac)
            st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
            
            show_data_overview(df)
            plot_basic_histogram(df)

            # You can add more plotting or analysis functions here
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
