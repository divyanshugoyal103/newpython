import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Pokemon Data Visualization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and preprocess Pokemon data"""
    # You'll need to replace this with your actual data loading method
    # df = pd.read_csv('pokemon.csv')
    # For demo purposes, we'll create a sample dataset structure
    # Replace this entire function with: df = pd.read_csv('pokemon.csv')

    # Sample data structure - replace with your actual data loading
    np.random.seed(42)
    n_pokemon = 801

    types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison',
             'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']

    df = pd.DataFrame({
        'name': [f'Pokemon_{i}' for i in range(1, n_pokemon + 1)],
        'type1': np.random.choice(types, n_pokemon),
        'type2': np.random.choice(types + [None], n_pokemon, p=[0.05]*18 + [0.1]),
        'generation': np.random.choice(range(1, 8), n_pokemon),
        'is_legendary': np.random.choice([0, 1], n_pokemon, p=[0.95, 0.05]),
        'hp': np.random.randint(20, 150, n_pokemon),
        'attack': np.random.randint(20, 150, n_pokemon),
        'defense': np.random.randint(20, 150, n_pokemon),
        'sp_attack': np.random.randint(20, 150, n_pokemon),
        'sp_defense': np.random.randint(20, 150, n_pokemon),
        'speed': np.random.randint(20, 150, n_pokemon),
        'height_m': np.random.uniform(0.1, 5.0, n_pokemon),
        'weight_kg': np.random.uniform(0.5, 200.0, n_pokemon),
        'capture_rate': np.random.randint(3, 255, n_pokemon),
        'base_happiness': np.random.randint(0, 100, n_pokemon),
    })

    # Calculate base total
    df['base_total'] = df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].sum(axis=1)

    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Pokemon Data Visualization Dashboard ⚡</h1>',
                unsafe_allow_html=True)

    # Load data
    df = load_data()

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Generation filter
    generations = sorted(df['generation'].unique())
    selected_generations = st.sidebar.multiselect(
        "Select Generation(s)",
        generations,
        default=generations
    )

    # Type filter
    types = sorted(df['type1'].unique())
    selected_types = st.sidebar.multiselect(
        "Select Primary Type(s)",
        types,
        default=types
    )

    # Legendary filter
    legendary_filter = st.sidebar.radio(
        "Pokemon Type",
        ["All", "Legendary Only", "Non-Legendary Only"]
    )

    # Apply filters
    filtered_df = df[
        (df['generation'].isin(selected_generations)) &
        (df['type1'].isin(selected_types))
    ]

    if legendary_filter == "Legendary Only":
        filtered_df = filtered_df[filtered_df['is_legendary'] == 1]
    elif legendary_filter == "Non-Legendary Only":
        filtered_df = filtered_df[filtered_df['is_legendary'] == 0]

    # Main dashboard
    if len(filtered_df) == 0:
        st.warning("No Pokemon match the selected filters. Please adjust your selection.")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Pokemon", len(filtered_df))
    with col2:
        st.metric("Legendary Pokemon", len(filtered_df[filtered_df['is_legendary'] == 1]))
    with col3:
        st.metric("Average Base Total", f"{filtered_df['base_total'].mean():.0f}")
    with col4:
        st.metric("Generations Covered", len(filtered_df['generation'].unique()))

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "💪 Stats Analysis", "🔥 Type Analysis",
        "📏 Physical Characteristics", "🎯 Advanced Analysis"
    ])

    with tab1:
        st.header("Dataset Overview")

        col1, col2 = st.columns(2)

        with col1:
            # Pokemon count by generation
            gen_counts = filtered_df['generation'].value_counts().sort_index()
            fig_gen = px.bar(
                x=gen_counts.index,
                y=gen_counts.values,
                title="Pokemon Count by Generation",
                labels={'x': 'Generation', 'y': 'Count'},
                color=gen_counts.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_gen, use_container_width=True)

        with col2:
            # Type distribution
            type_counts = filtered_df['type1'].value_counts()
            fig_type = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Primary Type Distribution"
            )
            st.plotly_chart(fig_type, use_container_width=True)

        # Legendary distribution
        legendary_counts = filtered_df['is_legendary'].value_counts()
        legendary_labels = ['Non-Legendary', 'Legendary']
        fig_legendary = px.pie(
            values=legendary_counts.values,
            names=[legendary_labels[i] for i in legendary_counts.index],
            title="Legendary vs Non-Legendary Distribution",
            color_discrete_map={'Legendary': '#FFD700', 'Non-Legendary': '#87CEEB'}
        )
        st.plotly_chart(fig_legendary, use_container_width=True)

    with tab2:
        st.header("Stats Analysis")

        # Stats correlation heatmap
        stats_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
        corr_matrix = filtered_df[stats_cols].corr()

        fig_heatmap = px.imshow(
            corr_matrix,
            title="Pokemon Stats Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Base total distribution
            fig_base_total = px.histogram(
                filtered_df,
                x='base_total',
                nbins=30,
                title="Base Total Stats Distribution",
                labels={'base_total': 'Base Total Stats'}
            )
            st.plotly_chart(fig_base_total, use_container_width=True)

        with col2:
            # Stats by generation
            stats_by_gen = filtered_df.groupby('generation')[stats_cols + ['base_total']].mean().reset_index()
            fig_gen_stats = px.line(
                stats_by_gen,
                x='generation',
                y='base_total',
                title="Average Base Total by Generation",
                markers=True
            )
            st.plotly_chart(fig_gen_stats, use_container_width=True)

        # Top Pokemon by base total
        st.subheader("Top 10 Pokemon by Base Total Stats")
        top_pokemon = filtered_df.nlargest(10, 'base_total')[
            ['name', 'type1', 'type2', 'base_total', 'generation', 'is_legendary']
        ]
        st.dataframe(top_pokemon, use_container_width=True)

    with tab3:
        st.header("Type Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Average stats by type
            type_stats = filtered_df.groupby('type1')['base_total'].mean().sort_values(ascending=False)
            fig_type_stats = px.bar(
                x=type_stats.index,
                y=type_stats.values,
                title="Average Base Total by Primary Type",
                labels={'x': 'Type', 'y': 'Average Base Total'}
            )
            fig_type_stats.update_xaxes(tickangle=45)
            st.plotly_chart(fig_type_stats, use_container_width=True)

        with col2:
            # Type combinations
            type_combo = filtered_df[filtered_df['type2'].notna()].groupby(['type1', 'type2']).size().reset_index(name='count')
            if len(type_combo) > 0:
                top_combos = type_combo.nlargest(10, 'count')
                top_combos['combination'] = top_combos['type1'] + ' + ' + top_combos['type2']

                fig_combo = px.bar(
                    top_combos,
                    x='combination',
                    y='count',
                    title="Top 10 Type Combinations"
                )
                fig_combo.update_xaxes(tickangle=45)
                st.plotly_chart(fig_combo, use_container_width=True)

        # Legendary distribution by type
        legendary_by_type = filtered_df.groupby('type1')['is_legendary'].agg(['count', 'sum']).reset_index()
        legendary_by_type['legendary_rate'] = legendary_by_type['sum'] / legendary_by_type['count'] * 100
        legendary_by_type = legendary_by_type.sort_values('legendary_rate', ascending=False)

        fig_legendary_type = px.bar(
            legendary_by_type,
            x='type1',
            y='legendary_rate',
            title="Legendary Rate by Primary Type (%)",
            labels={'type1': 'Type', 'legendary_rate': 'Legendary Rate (%)'}
        )
        fig_legendary_type.update_xaxes(tickangle=45)
        st.plotly_chart(fig_legendary_type, use_container_width=True)

    with tab4:
        st.header("Physical Characteristics")

        col1, col2 = st.columns(2)

        with col1:
            # Height vs Weight scatter
            fig_size = px.scatter(
                filtered_df,
                x='height_m',
                y='weight_kg',
                color='type1',
                size='base_total',
                hover_data=['name'],
                title="Height vs Weight (Size = Base Total)"
            )
            st.plotly_chart(fig_size, use_container_width=True)

        with col2:
            # Size vs Stats correlation
            fig_size_stats = px.scatter(
                filtered_df,
                x='weight_kg',
                y='base_total',
                color='is_legendary',
                hover_data=['name', 'type1'],
                title="Weight vs Base Total Stats"
            )
            st.plotly_chart(fig_size_stats, use_container_width=True)

        # Distribution of physical characteristics
        col1, col2 = st.columns(2)

        with col1:
            fig_height = px.histogram(
                filtered_df,
                x='height_m',
                nbins=30,
                title="Height Distribution"
            )
            st.plotly_chart(fig_height, use_container_width=True)

        with col2:
            fig_weight = px.histogram(
                filtered_df,
                x='weight_kg',
                nbins=30,
                title="Weight Distribution"
            )
            st.plotly_chart(fig_weight, use_container_width=True)

    with tab5:
        st.header("Advanced Analysis")

        # Radar chart for selected Pokemon
        st.subheader("Pokemon Stats Comparison")

        # Pokemon selector
        selected_pokemon = st.multiselect(
            "Select Pokemon to compare (max 5)",
            filtered_df['name'].tolist(),
            default=filtered_df.nlargest(3, 'base_total')['name'].tolist()
        )

        if selected_pokemon:
            selected_pokemon = selected_pokemon[:5]  # Limit to 5
            pokemon_data = filtered_df[filtered_df['name'].isin(selected_pokemon)]

            # Create radar chart
            fig_radar = go.Figure()

            stats_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

            for pokemon in selected_pokemon:
                pokemon_stats = pokemon_data[pokemon_data['name'] == pokemon][stats_cols].iloc[0]
                fig_radar.add_trace(go.Scatterpolar(
                    r=pokemon_stats.values,
                    theta=stats_cols,
                    fill='toself',
                    name=pokemon
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 150]
                    )),
                showlegend=True,
                title="Pokemon Stats Comparison (Radar Chart)"
            )

            st.plotly_chart(fig_radar, use_container_width=True)

        # Capture rate analysis
        col1, col2 = st.columns(2)

        with col1:
            # Capture rate vs base total
            fig_capture = px.scatter(
                filtered_df,
                x='capture_rate',
                y='base_total',
                color='is_legendary',
                title="Capture Rate vs Base Total Stats",
                labels={'capture_rate': 'Capture Rate', 'base_total': 'Base Total'}
            )
            st.plotly_chart(fig_capture, use_container_width=True)

        with col2:
            # Base happiness distribution
            fig_happiness = px.box(
                filtered_df,
                x='type1',
                y='base_happiness',
                title="Base Happiness by Type"
            )
            fig_happiness.update_xaxes(tickangle=45)
            st.plotly_chart(fig_happiness, use_container_width=True)

        # Data table
        st.subheader("Filtered Dataset")
        st.dataframe(filtered_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**Pokemon Data Visualization Dashboard** | Built with Streamlit & Plotly")

if __name__ == "__main__":
    main()