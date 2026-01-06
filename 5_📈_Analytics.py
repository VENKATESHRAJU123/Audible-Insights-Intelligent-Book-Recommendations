"""
Analytics Page
Comprehensive analytics and insights dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.components.visualization_panel import VisualizationPanel

# Page config
st.set_page_config(
    page_title="Analytics - Book Recommender",
    page_icon="📈",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load the book dataset"""
    try:
        df = pd.read_csv('data/processed/clustered_data.csv')
        
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower().str.strip()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def get_column_name(df, possible_names):
    """Get actual column name from list of possibilities"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def create_summary_metrics(df):
    """Create summary metrics cards"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold;">{len(df):,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Total Books</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        author_col = get_column_name(df, ['author', 'Author'])
        if author_col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{df[author_col].nunique():,}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Authors</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        genre_col = get_column_name(df, ['genre', 'Genre'])
        if genre_col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{df[genre_col].nunique():,}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Genres</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        rating_col = get_column_name(df, ['rating', 'Rating'])
        if rating_col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{df[rating_col].mean():.2f} ⭐</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Avg Rating</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        price_col = get_column_name(df, ['price', 'Price'])
        if price_col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">${df[price_col].mean():.2f}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Avg Price</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main analytics page function"""
    
    st.title("📈 Analytics & Insights Dashboard")
    st.markdown("### Explore comprehensive data analytics and visualizations")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("⚠️ No data available")
        return
    
    # Get column names (handle case variations)
    rating_col = get_column_name(df, ['rating', 'Rating'])
    price_col = get_column_name(df, ['price', 'Price'])
    genre_col = get_column_name(df, ['genre', 'Genre'])
    author_col = get_column_name(df, ['author', 'Author'])
    book_col = get_column_name(df, ['book_name', 'Book Name', 'book name'])
    
    # Initialize visualization panel
    viz_panel = VisualizationPanel()
    
    # Summary metrics
    st.markdown("## 📊 Key Metrics")
    create_summary_metrics(df)
    
    st.markdown("---")
    
    # Main analytics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🎭 Genres", "✍️ Authors", "💰 Pricing", "🎯 Advanced"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.markdown("## 📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            if rating_col:
                fig = px.histogram(
                    df,
                    x=rating_col,
                    nbins=30,
                    title="Rating Distribution",
                    labels={rating_col: 'Rating', 'count': 'Number of Books'},
                    color_discrete_sequence=['#4CAF50']
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Rating column not found")
        
        with col2:
            # Genre distribution (pie)
            if genre_col:
                genre_counts = df[genre_col].value_counts().head(10)
                fig = px.pie(
                    values=genre_counts.values,
                    names=genre_counts.index,
                    title="Genre Distribution (Top 10)"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Genre column not found")
        
        st.markdown("---")
        
        # Rating breakdown
        if rating_col:
            st.markdown("### ⭐ Rating Distribution Breakdown")
            
            rating_bins = pd.cut(df[rating_col], 
                                bins=[0, 2, 3, 4, 5],
                                labels=['Poor (0-2)', 'Fair (2-3)', 'Good (3-4)', 'Excellent (4-5)'])
            
            rating_counts = rating_bins.value_counts().sort_index()
            
            col1, col2, col3, col4 = st.columns(4)
            
            for col, (category, count) in zip([col1, col2, col3, col4], rating_counts.items()):
                percentage = (count / len(df)) * 100
                with col:
                    st.metric(
                        str(category),
                        f"{count:,} books",
                        f"{percentage:.1f}%"
                    )
    
    # TAB 2: Genre Analysis
    with tab2:
        st.markdown("## 🎭 Genre Deep Dive")
        
        if not genre_col:
            st.warning("Genre data not available")
        else:
            # Genre distribution
            genre_counts = df[genre_col].value_counts().head(15)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=genre_counts.values,
                    y=genre_counts.index,
                    orientation='h',
                    title="Top 15 Genres",
                    labels={'x': 'Number of Books', 'y': 'Genre'},
                    color=genre_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=genre_counts.values,
                    names=genre_counts.index,
                    title="Genre Distribution (Top 15)"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Genre statistics table
            st.markdown("### 📋 Genre Statistics")
            
            # Build aggregation dict dynamically
            agg_dict = {book_col: 'count'} if book_col else {'genre': 'count'}
            
            if rating_col:
                agg_dict[rating_col] = ['mean', 'std']
            
            if price_col:
                agg_dict[price_col] = 'mean'
            
            genre_stats = df.groupby(genre_col).agg(agg_dict).round(2)
            
            # Flatten column names
            genre_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                  for col in genre_stats.columns.values]
            
            genre_stats = genre_stats.sort_values(genre_stats.columns[0], ascending=False)
            
            # Interactive filter
            min_books = st.slider("Minimum books in genre", 1, 100, 10)
            filtered_stats = genre_stats[genre_stats.iloc[:, 0] >= min_books]
            
            st.dataframe(filtered_stats, use_container_width=True)
    
    # TAB 3: Author Analysis
    with tab3:
        st.markdown("## ✍️ Author Analytics")
        
        if not author_col:
            st.warning("Author data not available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Most Prolific Authors")
                author_counts = df[author_col].value_counts().head(20)
                
                fig = px.bar(
                    x=author_counts.values,
                    y=author_counts.index,
                    orientation='h',
                    title="Top 20 Most Prolific Authors",
                    labels={'x': 'Number of Books', 'y': 'Author'}
                )
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Highest Rated Authors")
                
                if rating_col:
                    author_ratings = df.groupby(author_col).agg({
                        rating_col: ['mean', 'count']
                    }).reset_index()
                    author_ratings.columns = ['author', 'avg_rating', 'book_count']
                    
                    # Filter authors with at least 3 books
                    top_rated = author_ratings[
                        author_ratings['book_count'] >= 3
                    ].sort_values('avg_rating', ascending=False).head(20)
                    
                    fig = px.bar(
                        x=top_rated['avg_rating'],
                        y=top_rated['author'],
                        orientation='h',
                        title="Top 20 Highest Rated Authors (min 3 books)",
                        labels={'x': 'Average Rating', 'y': 'Author'}
                    )
                    fig.update_layout(showlegend=False, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Rating data not available")
    
    # TAB 4: Pricing Analysis
    with tab4:
        st.markdown("## 💰 Price Analytics")
        
        if not price_col:
            st.warning("Price data not available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution
                fig = px.histogram(
                    df,
                    x=price_col,
                    nbins=40,
                    title="Price Distribution",
                    labels={price_col: 'Price ($)', 'count': 'Number of Books'}
                )
                
                mean_price = df[price_col].mean()
                fig.add_vline(
                    x=mean_price,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: ${mean_price:.2f}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price statistics
                st.markdown("### 💵 Price Statistics")
                
                price_stats = df[price_col].describe()
                
                for stat, value in price_stats.items():
                    st.metric(stat.capitalize(), f"${value:.2f}")
            
            st.markdown("---")
            
            # Price vs Rating
            if rating_col:
                st.markdown("### 📊 Price vs Rating Analysis")
                
                sample_df = df.sample(min(1000, len(df)))
                
                fig = px.scatter(
                    sample_df,
                    x=price_col,
                    y=rating_col,
                    title="Price vs Rating Scatter Plot",
                    labels={price_col: 'Price ($)', rating_col: 'Rating'},
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: Advanced Analytics
    with tab5:
        st.markdown("## 🎯 Advanced Analytics")
        
        # Correlation analysis
        st.markdown("### 🔗 Feature Correlations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu',
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for correlation analysis")
        
        st.markdown("---")
        
        # Custom analysis
        st.markdown("### 🔬 Custom Analysis")
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis", numeric_cols, key="x_axis")
            
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols, 
                                     index=1 if len(numeric_cols) > 1 else 0, 
                                     key="y_axis")
            
            if x_axis and y_axis:
                color_col = genre_col if genre_col else None
                
                sample_size = min(1000, len(df))
                sample_df = df.sample(sample_size)
                
                hover_data = []
                if book_col:
                    hover_data.append(book_col)
                if author_col:
                    hover_data.append(author_col)
                
                fig = px.scatter(
                    sample_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_col,
                    title=f"{x_axis} vs {y_axis} (Sample: {sample_size} books)",
                    hover_data=hover_data if hover_data else None
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Export section
    st.markdown("---")
    st.markdown("## 📥 Export Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Full Dataset", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "full_dataset.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("📈 Export Summary Stats", use_container_width=True):
            summary = df.describe()
            csv = summary.to_csv()
            st.download_button(
                "Download Summary",
                csv,
                "summary_statistics.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.button("📄 Generate Report", use_container_width=True):
            st.info("Comprehensive report generation coming soon!")


if __name__ == "__main__":
    main()
