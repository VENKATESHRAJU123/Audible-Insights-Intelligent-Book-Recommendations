"""
Visualization Panel Component
Creates interactive visualizations for book data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List


class VisualizationPanel:
    """Component for creating data visualizations"""
    
    def __init__(self):
        """Initialize VisualizationPanel"""
        self.default_colors = px.colors.qualitative.Set3
    
    
    def plot_rating_distribution(self, df: pd.DataFrame, 
                                title: str = "Rating Distribution"):
        """
        Plot rating distribution histogram
        
        Args:
            df: DataFrame with rating column
            title: Plot title
        """
        if 'rating' not in df.columns:
            st.warning("Rating data not available")
            return
        
        fig = px.histogram(
            df,
            x='rating',
            nbins=30,
            title=title,
            labels={'rating': 'Rating', 'count': 'Number of Books'},
            color_discrete_sequence=['#4CAF50']
        )
        
        # Add mean and median lines
        mean_rating = df['rating'].mean()
        median_rating = df['rating'].median()
        
        fig.add_vline(
            x=mean_rating, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {mean_rating:.2f}",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=median_rating, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Median: {median_rating:.2f}",
            annotation_position="bottom"
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    
    def plot_genre_distribution(self, df: pd.DataFrame, 
                               top_n: int = 10,
                               chart_type: str = "bar"):
        """
        Plot genre distribution
        
        Args:
            df: DataFrame with genre column
            top_n: Number of top genres to show
            chart_type: 'bar', 'pie', or 'both'
        """
        if 'genre' not in df.columns:
            st.warning("Genre data not available")
            return
        
        genre_counts = df['genre'].value_counts().head(top_n)
        
        if chart_type in ["bar", "both"]:
            fig_bar = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title=f"Top {top_n} Genres",
                labels={'x': 'Number of Books', 'y': 'Genre'},
                color=genre_counts.values,
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(
                showlegend=False,
                height=400
            )
            
            if chart_type == "both":
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.plotly_chart(fig_bar, use_container_width=True)
        
        if chart_type in ["pie", "both"]:
            fig_pie = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title=f"Genre Distribution (Top {top_n})"
            )
            fig_pie.update_layout(height=400)
            
            if chart_type == "both":
                with col2:
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.plotly_chart(fig_pie, use_container_width=True)
    
    
    def plot_price_vs_rating(self, df: pd.DataFrame, 
                            sample_size: int = 1000):
        """
        Plot price vs rating scatter plot
        
        Args:
            df: DataFrame with price and rating columns
            sample_size: Number of samples to plot
        """
        if 'price' not in df.columns or 'rating' not in df.columns:
            st.warning("Price or rating data not available")
            return
        
        # Sample for performance
        sample_df = df.sample(min(sample_size, len(df)))
        
        fig = px.scatter(
            sample_df,
            x='price',
            y='rating',
            color='genre' if 'genre' in sample_df.columns else None,
            size='price',
            hover_data=['book_name', 'author'] if 'author' in sample_df.columns else ['book_name'],
            title=f"Price vs Rating (Sample: {len(sample_df)} books)",
            labels={'price': 'Price ($)', 'rating': 'Rating'},
            trendline="ols",
            trendline_color_override="red"
        )
        
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation
        corr = sample_df[['price', 'rating']].corr().iloc[0, 1]
        st.caption(f"📊 Correlation coefficient: {corr:.3f}")
    
    
    def plot_author_analysis(self, df: pd.DataFrame, 
                            top_n: int = 15,
                            metric: str = "count"):
        """
        Plot author analysis
        
        Args:
            df: DataFrame with author column
            top_n: Number of top authors to show
            metric: 'count' or 'rating'
        """
        if 'author' not in df.columns:
            st.warning("Author data not available")
            return
        
        if metric == "count":
            author_data = df['author'].value_counts().head(top_n)
            title = f"Top {top_n} Most Prolific Authors"
            x_label = "Number of Books"
        else:  # rating
            if 'rating' not in df.columns:
                st.warning("Rating data not available")
                return
            
            author_ratings = df.groupby('author').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            author_ratings.columns = ['author', 'avg_rating', 'book_count']
            
            # Filter authors with at least 3 books
            author_ratings = author_ratings[
                author_ratings['book_count'] >= 3
            ].sort_values('avg_rating', ascending=False).head(top_n)
            
            author_data = author_ratings.set_index('author')['avg_rating']
            title = f"Top {top_n} Highest Rated Authors (min 3 books)"
            x_label = "Average Rating"
        
        fig = px.bar(
            x=author_data.values,
            y=author_data.index,
            orientation='h',
            title=title,
            labels={'x': x_label, 'y': 'Author'}
        )
        
        fig.update_layout(
            showlegend=False,
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    
    def plot_cluster_analysis(self, df: pd.DataFrame):
        """
        Plot cluster analysis
        
        Args:
            df: DataFrame with cluster column
        """
        if 'cluster_kmeans' not in df.columns:
            st.warning("Cluster data not available")
            return
        
        # Cluster distribution
        cluster_counts = df['cluster_kmeans'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                marker=dict(
                    color=cluster_counts.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Books")
                )
            )
        ])
        
        fig.update_layout(
            title="Books per Cluster",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Books",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        if 'rating' in df.columns and 'genre' in df.columns:
            st.markdown("### Cluster Characteristics")
            
            cluster_stats = df.groupby('cluster_kmeans').agg({
                'rating': 'mean',
                'genre': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
            }).round(2)
            
            cluster_stats.columns = ['Avg Rating', 'Top Genre']
            st.dataframe(cluster_stats, use_container_width=True)
    
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """
        Plot correlation heatmap for numeric features
        
        Args:
            df: DataFrame with numeric columns
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns for correlation analysis")
            return
        
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
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    
    def create_summary_dashboard(self, df: pd.DataFrame):
        """
        Create a comprehensive summary dashboard
        
        Args:
            df: DataFrame with book data
        """
        st.markdown("## 📊 Summary Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Books", f"{len(df):,}")
        
        with col2:
            if 'author' in df.columns:
                st.metric("✍️ Authors", f"{df['author'].nunique():,}")
        
        with col3:
            if 'genre' in df.columns:
                st.metric("🎭 Genres", f"{df['genre'].nunique():,}")
        
        with col4:
            if 'rating' in df.columns:
                st.metric("⭐ Avg Rating", f"{df['rating'].mean():.2f}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_rating_distribution(df)
        
        with col2:
            self.plot_genre_distribution(df, top_n=10, chart_type="bar")
