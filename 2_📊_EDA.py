"""
Exploratory Data Analysis Page
Detailed data exploration and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.components.visualization_panel import VisualizationPanel

# Page config
st.set_page_config(
    page_title="EDA - Book Recommender",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load the book dataset"""
    try:
        df = pd.read_csv('data/processed/clustered_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def main():
    """Main EDA page function"""
    
    st.title("📊 Exploratory Data Analysis")
    st.markdown("### Deep dive into the book dataset")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("⚠️ No data available")
        return
    
    # Initialize visualization panel
    viz_panel = VisualizationPanel()
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Overview", "📈 Distributions", "🔗 Correlations", 
        "🎭 Genre Deep Dive", "✍️ Author Analysis", "🎯 Clustering"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.markdown("## 📋 Dataset Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Basic Information")
            
            info_data = {
                "Metric": [
                    "Total Records",
                    "Total Columns",
                    "Numeric Columns",
                    "Categorical Columns",
                    "Missing Values",
                    "Duplicate Rows"
                ],
                "Value": [
                    f"{len(df):,}",
                    len(df.columns),
                    len(df.select_dtypes(include=[np.number]).columns),
                    len(df.select_dtypes(include=['object']).columns),
                    f"{df.isnull().sum().sum():,}",
                    f"{df.duplicated().sum():,}"
                ]
            }
            
            st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### Dataset Metrics")
            
            if 'book_name' in df.columns:
                st.metric("Unique Books", f"{df['book_name'].nunique():,}")
            
            if 'author' in df.columns:
                st.metric("Unique Authors", f"{df['author'].nunique():,}")
            
            if 'genre' in df.columns:
                st.metric("Unique Genres", f"{df['genre'].nunique():,}")
        
        st.markdown("---")
        
        # Column details
        st.markdown("### Column Details")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Sample data
        st.markdown("### Sample Data")
        
        num_samples = st.slider("Number of samples to display", 5, 50, 10)
        st.dataframe(df.sample(num_samples), use_container_width=True)
    
    # TAB 2: Distributions
    with tab2:
        st.markdown("## 📈 Data Distributions")
        
        # Rating distribution
        if 'rating' in df.columns:
            st.markdown("### ⭐ Rating Distribution")
            viz_panel.plot_rating_distribution(df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Rating", f"{df['rating'].mean():.2f}")
            with col2:
                st.metric("Median Rating", f"{df['rating'].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df['rating'].std():.2f}")
            with col4:
                st.metric("Mode", f"{df['rating'].mode()[0]:.2f}")
        
        st.markdown("---")
        
        # Price distribution
        if 'price' in df.columns:
            st.markdown("### 💰 Price Distribution")
            
            import plotly.express as px
            
            # Remove extreme outliers for better visualization
            q1 = df['price'].quantile(0.25)
            q3 = df['price'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            price_filtered = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
            
            fig = px.histogram(
                price_filtered,
                x='price',
                nbins=40,
                title="Price Distribution (Outliers Removed)",
                labels={'price': 'Price ($)', 'count': 'Frequency'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Price", f"${df['price'].mean():.2f}")
            with col2:
                st.metric("Median Price", f"${df['price'].median():.2f}")
            with col3:
                st.metric("Min Price", f"${df['price'].min():.2f}")
            with col4:
                st.metric("Max Price", f"${df['price'].max():.2f}")
        
        st.markdown("---")
        
        # Review distribution
        review_cols = [col for col in df.columns if 'review' in col.lower()]
        if review_cols:
            st.markdown("### 📝 Review Count Distribution")
            
            review_col = review_cols[0]
            
            fig = px.histogram(
                df,
                x=review_col,
                nbins=30,
                title=f"Distribution of {review_col}",
                labels={review_col: 'Number of Reviews', 'count': 'Frequency'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Correlations
    with tab3:
        st.markdown("## 🔗 Feature Correlations")
        
        viz_panel.plot_correlation_heatmap(df)
        
        st.markdown("---")
        
        # Specific correlations
        if 'rating' in df.columns and 'price' in df.columns:
            st.markdown("### Price vs Rating Analysis")
            viz_panel.plot_price_vs_rating(df, sample_size=1000)
    
    # TAB 4: Genre Analysis
    with tab4:
        st.markdown("## 🎭 Genre Deep Dive")
        
        if 'genre' not in df.columns:
            st.warning("Genre data not available")
        else:
            # Genre distribution
            viz_panel.plot_genre_distribution(df, top_n=15, chart_type="both")
            
            st.markdown("---")
            
            # Genre statistics
            st.markdown("### Genre Statistics")
            
            genre_stats = df.groupby('genre').agg({
                'book_name': 'count',
                'rating': ['mean', 'median', 'std'] if 'rating' in df.columns else 'count',
                'price': ['mean', 'median'] if 'price' in df.columns else 'count'
            }).round(2)
            
            genre_stats.columns = ['_'.join(col).strip('_') for col in genre_stats.columns.values]
            genre_stats = genre_stats.sort_values(genre_stats.columns[0], ascending=False)
            
            st.dataframe(genre_stats.head(20), use_container_width=True)
            
            # Download option
            csv = genre_stats.to_csv()
            st.download_button(
                label="📥 Download Genre Statistics",
                data=csv,
                file_name="genre_statistics.csv",
                mime="text/csv"
            )
    
    # TAB 5: Author Analysis
    with tab5:
        st.markdown("## ✍️ Author Analysis")
        
        if 'author' not in df.columns:
            st.warning("Author data not available")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Most Prolific Authors")
                viz_panel.plot_author_analysis(df, top_n=15, metric="count")
            
            with col2:
                st.markdown("### Highest Rated Authors")
                viz_panel.plot_author_analysis(df, top_n=15, metric="rating")
            
            st.markdown("---")
            
            # Author statistics table
            st.markdown("### Author Statistics")
            
            if 'rating' in df.columns:
                author_stats = df.groupby('author').agg({
                    'book_name': 'count',
                    'rating': ['mean', 'median', 'std'],
                    'price': 'mean' if 'price' in df.columns else 'count'
                }).round(2)
                
                author_stats.columns = ['Books', 'Avg_Rating', 'Median_Rating', 
                                       'Rating_Std', 'Avg_Price' if 'price' in df.columns else 'Price']
                author_stats = author_stats.sort_values('Books', ascending=False)
                
                # Filter
                min_books = st.slider("Minimum books by author", 1, 10, 3)
                filtered_authors = author_stats[author_stats['Books'] >= min_books]
                
                st.dataframe(filtered_authors.head(50), use_container_width=True)
    
    # TAB 6: Clustering
    with tab6:
        st.markdown("## 🎯 Clustering Analysis")
        
        if 'cluster_kmeans' not in df.columns:
            st.warning("Clustering data not available")
        else:
            viz_panel.plot_cluster_analysis(df)
            
            st.markdown("---")
            
            # Cluster exploration
            st.markdown("### Explore Clusters")
            
            cluster_id = st.selectbox(
                "Select Cluster",
                sorted(df['cluster_kmeans'].unique())
            )
            
            cluster_books = df[df['cluster_kmeans'] == cluster_id]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Books in Cluster", len(cluster_books))
            
            with col2:
                if 'rating' in cluster_books.columns:
                    st.metric("Avg Rating", f"{cluster_books['rating'].mean():.2f}")
            
            with col3:
                if 'genre' in cluster_books.columns:
                    top_genre = cluster_books['genre'].mode()[0] if not cluster_books['genre'].mode().empty else 'N/A'
                    st.metric("Top Genre", top_genre)
            
            # Sample books from cluster
            st.markdown(f"#### Sample Books from Cluster {cluster_id}")
            st.dataframe(
                cluster_books[['book_name', 'author', 'genre', 'rating']].head(10),
                use_container_width=True,
                hide_index=True
            )


if __name__ == "__main__":
    main()
