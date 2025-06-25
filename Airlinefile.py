import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="✈️ Airline Analytics Dashboard",
    page_icon="✈️",
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
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the airline dataset"""
    try:
        # Update this path to your dataset location
        df = pd.read_csv('Airline Dataset Updated - v2.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert departure date to datetime
        df['Departure Date'] = pd.to_datetime(df['Departure Date'], errors='coerce')
        
        # Extract date components
        df['Year'] = df['Departure Date'].dt.year
        df['Month'] = df['Departure Date'].dt.month
        df['Month_Name'] = df['Departure Date'].dt.month_name()
        df['Day_of_Week'] = df['Departure Date'].dt.day_name()
        df['Quarter'] = df['Departure Date'].dt.quarter
        
        # Create age groups
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 18, 30, 45, 60, 100], 
                                labels=['<18', '18-30', '31-45', '46-60', '60+'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.markdown('<h1 class="main-header">✈️ Airline Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Please upload your 'Airline Dataset Updated v2.csv' file to the same directory as this script.")
        return
    
    # Sidebar filters
    st.sidebar.title("🔍 Filters")
    
    # Date range filter
    if not df['Departure Date'].isna().all():
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['Departure Date'].min().date(), df['Departure Date'].max().date()),
            min_value=df['Departure Date'].min().date(),
            max_value=df['Departure Date'].max().date()
        )
        
        # Filter data by date range
        if len(date_range) == 2:
            df_filtered = df[
                (df['Departure Date'].dt.date >= date_range[0]) & 
                (df['Departure Date'].dt.date <= date_range[1])
            ]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    # Other filters
    continents = st.sidebar.multiselect(
        "Select Continents",
        options=df['Continents'].unique(),
        default=df['Continents'].unique()
    )
    
    flight_status = st.sidebar.multiselect(
        "Select Flight Status",
        options=df['Flight Status'].unique(),
        default=df['Flight Status'].unique()
    )
    
    gender = st.sidebar.multiselect(
        "Select Gender",
        options=df['Gender'].unique(),
        default=df['Gender'].unique()
    )
    
    # Apply filters
    df_filtered = df_filtered[
        (df_filtered['Continents'].isin(continents)) &
        (df_filtered['Flight Status'].isin(flight_status)) &
        (df_filtered['Gender'].isin(gender))
    ]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🌍 Geographic Analysis", "👥 Passenger Demographics", 
        "🕐 Time Analysis", "✈️ Flight Operations"
    ])
    
    with tab1:
        st.header("📊 Key Metrics Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Passengers", f"{len(df_filtered):,}")
        
        with col2:
            avg_age = df_filtered['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        
        with col3:
            unique_countries = df_filtered['Country Name'].nunique()
            st.metric("Countries Served", unique_countries)
        
        with col4:
            on_time_rate = (df_filtered['Flight Status'] == 'On Time').mean() * 100
            st.metric("On-Time Rate", f"{on_time_rate:.1f}%")
        
        with col5:
            unique_airports = df_filtered['Airport Name'].nunique()
            st.metric("Airports", unique_airports)
        
        st.markdown("---")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Flight Status Distribution")
            status_counts = df_filtered['Flight Status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Flight Status Breakdown",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        with col2:
            st.subheader("Gender Distribution")
            gender_counts = df_filtered['Gender'].value_counts()
            fig_gender = px.bar(
                x=gender_counts.index,
                y=gender_counts.values,
                title="Passengers by Gender",
                color=gender_counts.index,
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            fig_gender.update_layout(showlegend=False)
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig_age = px.histogram(
                df_filtered,
                x='Age',
                nbins=30,
                title="Age Distribution of Passengers",
                color_discrete_sequence=['#74B9FF']
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Nationalities")
            top_nationalities = df_filtered['Nationality'].value_counts().head(10)
            fig_nat = px.bar(
                x=top_nationalities.values,
                y=top_nationalities.index,
                orientation='h',
                title="Most Common Passenger Nationalities",
                color_discrete_sequence=['#A29BFE']
            )
            fig_nat.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig_nat, use_container_width=True)
    
    with tab2:
        st.header("🌍 Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Passengers by Continent")
            continent_counts = df_filtered['Continents'].value_counts()
            fig_cont = px.bar(
                x=continent_counts.index,
                y=continent_counts.values,
                title="Passenger Distribution by Continent",
                color=continent_counts.values,
                color_continuous_scale='Blues'
            )
            fig_cont.update_xaxes(tickangle=45)
            st.plotly_chart(fig_cont, use_container_width=True)
        
        with col2:
            st.subheader("Top Departure Countries")
            country_counts = df_filtered['Country Name'].value_counts().head(15)
            fig_countries = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                title="Top 15 Departure Countries",
                color_discrete_sequence=['#00B894']
            )
            fig_countries.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig_countries, use_container_width=True)
        
        # Continental breakdown
        st.subheader("Continental Flight Status Analysis")
        cont_status = pd.crosstab(df_filtered['Continents'], df_filtered['Flight Status'])
        fig_cont_status = px.bar(
            cont_status,
            title="Flight Status by Continent",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_cont_status, use_container_width=True)
        
        # Airport analysis
        st.subheader("Busiest Airports")
        airport_counts = df_filtered['Airport Name'].value_counts().head(20)
        fig_airports = px.bar(
            x=airport_counts.values,
            y=airport_counts.index,
            orientation='h',
            title="Top 20 Busiest Airports",
            color=airport_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_airports.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_airports, use_container_width=True)
    
    with tab3:
        st.header("👥 Passenger Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Groups Distribution")
            age_group_counts = df_filtered['Age_Group'].value_counts()
            fig_age_groups = px.pie(
                values=age_group_counts.values,
                names=age_group_counts.index,
                title="Passengers by Age Group",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_age_groups, use_container_width=True)
        
        with col2:
            st.subheader("Gender vs Age Analysis")
            fig_gender_age = px.box(
                df_filtered,
                x='Gender',
                y='Age',
                title="Age Distribution by Gender",
                color='Gender',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            st.plotly_chart(fig_gender_age, use_container_width=True)
        
        # Demographics by continent
        st.subheader("Demographics by Continent")
        demo_continent = df_filtered.groupby(['Continents', 'Gender']).size().reset_index(name='Count')
        fig_demo_cont = px.bar(
            demo_continent,
            x='Continents',
            y='Count',
            color='Gender',
            title="Gender Distribution Across Continents",
            barmode='group'
        )
        st.plotly_chart(fig_demo_cont, use_container_width=True)
        
        # Age statistics by continent
        st.subheader("Age Statistics by Continent")
        age_stats = df_filtered.groupby('Continents')['Age'].agg(['mean', 'median', 'std']).round(2)
        st.dataframe(age_stats, use_container_width=True)
    
    with tab4:
        st.header("🕐 Time Analysis")
        
        if not df_filtered['Departure Date'].isna().all():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Flights by Month")
                monthly_flights = df_filtered['Month_Name'].value_counts()
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                monthly_flights = monthly_flights.reindex([m for m in month_order if m in monthly_flights.index])
                
                fig_monthly = px.line(
                    x=monthly_flights.index,
                    y=monthly_flights.values,
                    title="Monthly Flight Volume",
                    markers=True
                )
                fig_monthly.update_xaxes(tickangle=45)
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with col2:
                st.subheader("Flights by Day of Week")
                daily_flights = df_filtered['Day_of_Week'].value_counts()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_flights = daily_flights.reindex([d for d in day_order if d in daily_flights.index])
                
                fig_daily = px.bar(
                    x=daily_flights.index,
                    y=daily_flights.values,
                    title="Flight Volume by Day of Week",
                    color=daily_flights.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            
            # Quarterly analysis
            st.subheader("Quarterly Flight Analysis")
            quarterly_data = df_filtered.groupby(['Quarter', 'Flight Status']).size().reset_index(name='Count')
            fig_quarterly = px.bar(
                quarterly_data,
                x='Quarter',
                y='Count',
                color='Flight Status',
                title="Flight Status by Quarter",
                barmode='group'
            )
            st.plotly_chart(fig_quarterly, use_container_width=True)
            
            # Heatmap of flights by month and day of week
            st.subheader("Flight Density Heatmap")
            heatmap_data = df_filtered.groupby(['Month_Name', 'Day_of_Week']).size().reset_index(name='Count')
            heatmap_pivot = heatmap_data.pivot(index='Day_of_Week', columns='Month_Name', values='Count').fillna(0)
            
            fig_heatmap = px.imshow(
                heatmap_pivot,
                title="Flight Volume Heatmap (Day of Week vs Month)",
                color_continuous_scale='Blues',
                aspect='auto'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab5:
        st.header("✈️ Flight Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Pilots by Flight Count")
            pilot_counts = df_filtered['Pilot Name'].value_counts().head(15)
            fig_pilots = px.bar(
                x=pilot_counts.values,
                y=pilot_counts.index,
                orientation='h',
                title="Most Active Pilots",
                color_discrete_sequence=['#6C5CE7']
            )
            fig_pilots.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig_pilots, use_container_width=True)
        
        with col2:
            st.subheader("Arrival Airports")
            arrival_counts = df_filtered['Arrival Airport'].value_counts().head(15)
            fig_arrivals = px.bar(
                x=arrival_counts.index,
                y=arrival_counts.values,
                title="Top Arrival Airports",
                color=arrival_counts.values,
                color_continuous_scale='Oranges'
            )
            fig_arrivals.update_xaxes(tickangle=45)
            st.plotly_chart(fig_arrivals, use_container_width=True)
        
        # Flight status by pilot performance
        st.subheader("Pilot Performance Analysis")
        pilot_performance = df_filtered.groupby('Pilot Name')['Flight Status'].apply(
            lambda x: (x == 'On Time').mean() * 100
        ).sort_values(ascending=False).head(20)
        
        fig_pilot_perf = px.bar(
            x=pilot_performance.index,
            y=pilot_performance.values,
            title="Top 20 Pilots by On-Time Performance (%)",
            color=pilot_performance.values,
            color_continuous_scale='RdYlGn'
        )
        fig_pilot_perf.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pilot_perf, use_container_width=True)
        
        # Operations summary table
        st.subheader("Operations Summary")
        ops_summary = df_filtered.groupby('Airport Name').agg({
            'Passenger ID': 'count',
            'Age': 'mean',
            'Flight Status': lambda x: (x == 'On Time').mean() * 100
        }).round(2)
        ops_summary.columns = ['Total Passengers', 'Avg Age', 'On-Time Rate (%)']
        ops_summary = ops_summary.sort_values('Total Passengers', ascending=False).head(20)
        
        st.dataframe(ops_summary, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            ✈️ Airline Analytics Dashboard | Built with Streamlit & Plotly
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
