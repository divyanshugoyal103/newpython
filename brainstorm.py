import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(layout="wide")
st.title("📊 Descriptive Data Analysis Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        # ---------- 🧹 Data Cleaning ----------
        st.sidebar.header("🧹 Data Cleaning")

        if st.sidebar.checkbox("Make column names lowercase"):
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        if st.sidebar.checkbox("Drop duplicates"):
            df.drop_duplicates(inplace=True)

        na_action = st.sidebar.selectbox("Handle missing values", ["Do nothing", "Drop rows", "Fill with 0"])
        if na_action == "Drop rows":
            df.dropna(inplace=True)
        elif na_action == "Fill with 0":
            df.fillna(0, inplace=True)

        if st.sidebar.checkbox("Try to convert date columns"):
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    continue

        # ---------- 📊 Filter Widgets ----------
        st.sidebar.header("🔍 Filters")
        filterable_cols = df.select_dtypes(include=["object", "category", "datetime64[ns]"]).columns.tolist()
        for col in filterable_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) < 50:
                selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
                df = df[df[col].isin(selected_vals)]

        # ---------- 🧮 KPIs ----------
        st.subheader("📌 Key Stats")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", df.shape[0])
            st.metric("Total Columns", df.shape[1])

        with col2:
            na_pct = (df.isnull().sum().sum() / df.size) * 100
            st.metric("% Missing Values", f"{na_pct:.2f}%")

        with col3:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            most_common = df[cat_cols[0]].value_counts().idxmax() if len(cat_cols) > 0 else "N/A"
            st.metric("Most Common Category", most_common)

        # ---------- 📄 Preview & Download ----------
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")

        # ---------- 📈 Plot Builder ----------
        st.subheader("📊 Create a Plot")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

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
            color_col = st.selectbox("Color", cat_cols + [None], index=0)
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col)

        st.plotly_chart(fig, use_container_width=True)

        # ---------- 📊 Grouped Summary Table ----------
        st.subheader("📂 Grouped Summary Table")
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            group_by_col = st.selectbox("Group by column", cat_cols)
            agg_col = st.selectbox("Aggregate column", numeric_cols)
            summary_df = df.groupby(group_by_col)[agg_col].agg(['count', 'mean', 'sum']).reset_index()
            st.dataframe(summary_df)

        # ---------- 🔥 Correlation Heatmap ----------
        st.subheader("🔗 Correlation Heatmap")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = ff.create_annotated_heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                annotation_text=corr.round(2).astype(str).values,
                showscale=True,
                colorscale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("Upload a CSV or Excel file to begin analysis.")
