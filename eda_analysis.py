"""
Exploratory Data Analysis Module
Comprehensive EDA functions for book recommendation system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Comprehensive EDA analyzer for book datasets"""
    
    def __init__(self, processed_data_path: str = "data/processed",
                 output_path: str = "outputs/visualizations"):
        """
        Initialize EDA Analyzer
        
        Args:
            processed_data_path: Path to processed data
            output_path: Path to save visualizations
        """
        self.processed_data_path = Path(processed_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        logger.info("EDAAnalyzer initialized")
    
    
    def load_data(self, filename: str = "cleaned_data.csv") -> pd.DataFrame:
        """
        Load cleaned dataset
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading data from {filename}...")
        
        filepath = self.processed_data_path / filename
        self.df = pd.read_csv(filepath, encoding='utf-8')
        
        logger.info(f"Data loaded: {self.df.shape}")
        return self.df
    
    
    def generate_overview_report(self) -> Dict:
        """
        Generate comprehensive overview report
        
        Returns:
            Dictionary with overview statistics
        """
        logger.info("Generating overview report...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        report = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 ** 2),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns),
        }
        
        # Add column-specific info
        if 'book_name' in self.df.columns:
            report['unique_books'] = self.df['book_name'].nunique()
        
        if 'author' in self.df.columns:
            report['unique_authors'] = self.df['author'].nunique()
        
        if 'genre' in self.df.columns:
            report['unique_genres'] = self.df['genre'].nunique()
        
        if 'rating' in self.df.columns:
            report['avg_rating'] = self.df['rating'].mean()
            report['median_rating'] = self.df['rating'].median()
        
        logger.info("Overview report generated")
        return report
    
    
    def print_overview(self):
        """Print formatted overview report"""
        report = self.generate_overview_report()
        
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        print(f"Total Records:        {report['total_records']:,}")
        print(f"Total Columns:        {report['total_columns']}")
        print(f"Memory Usage:         {report['memory_usage_mb']:.2f} MB")
        print(f"Missing Values:       {report['missing_values']:,}")
        print(f"Duplicate Rows:       {report['duplicate_rows']:,}")
        print(f"Numeric Columns:      {report['numeric_columns']}")
        print(f"Categorical Columns:  {report['categorical_columns']}")
        
        if 'unique_books' in report:
            print(f"\nUnique Books:         {report['unique_books']:,}")
        if 'unique_authors' in report:
            print(f"Unique Authors:       {report['unique_authors']:,}")
        if 'unique_genres' in report:
            print(f"Unique Genres:        {report['unique_genres']:,}")
        if 'avg_rating' in report:
            print(f"\nAverage Rating:       {report['avg_rating']:.2f}")
            print(f"Median Rating:        {report['median_rating']:.2f}")
        
        print("="*60 + "\n")
    
    
    def analyze_ratings(self, save_plot: bool = True) -> Dict:
        """
        Analyze rating distribution and statistics
        
        Args:
            save_plot: Whether to save the plot
            
        Returns:
            Dictionary with rating statistics
        """
        logger.info("Analyzing ratings...")
        
        if 'rating' not in self.df.columns:
            logger.warning("Rating column not found")
            return {}
        
        stats = {
            'mean': self.df['rating'].mean(),
            'median': self.df['rating'].median(),
            'std': self.df['rating'].std(),
            'min': self.df['rating'].min(),
            'max': self.df['rating'].max(),
            'q1': self.df['rating'].quantile(0.25),
            'q3': self.df['rating'].quantile(0.75)
        }
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(self.df['rating'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.2f}")
        axes[0, 0].axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.2f}")
        axes[0, 0].set_xlabel('Rating', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(self.df['rating'].dropna(), vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0, 1].set_ylabel('Rating', fontsize=12)
        axes[0, 1].set_title('Rating Box Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Rating categories
        rating_categories = pd.cut(self.df['rating'], 
                                   bins=[0, 2, 3, 4, 5], 
                                   labels=['Poor (0-2)', 'Fair (2-3)', 'Good (3-4)', 'Excellent (4-5)'])
        category_counts = rating_categories.value_counts().sort_index()
        
        axes[1, 0].bar(range(len(category_counts)), category_counts.values, 
                      color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_xticks(range(len(category_counts)))
        axes[1, 0].set_xticklabels(category_counts.index, rotation=15, ha='right')
        axes[1, 0].set_ylabel('Number of Books', fontsize=12)
        axes[1, 0].set_title('Books by Rating Category', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(category_counts.values):
            axes[1, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        # Density plot
        self.df['rating'].plot(kind='density', ax=axes[1, 1], color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Rating', fontsize=12)
        axes[1, 1].set_ylabel('Density', fontsize=12)
        axes[1, 1].set_title('Rating Density Plot', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path / 'rating_analysis.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Rating analysis plot saved to: {output_file}")
        
        plt.show()
        
        return stats
    
    
    def analyze_genres(self, top_n: int = 15, save_plot: bool = True) -> pd.Series:
        """
        Analyze genre distribution
        
        Args:
            top_n: Number of top genres to analyze
            save_plot: Whether to save the plot
            
        Returns:
            Series with genre counts
        """
        logger.info("Analyzing genres...")
        
        if 'genre' not in self.df.columns:
            logger.warning("Genre column not found")
            return pd.Series()
        
        genre_counts = self.df['genre'].value_counts().head(top_n)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Horizontal bar chart
        colors = sns.color_palette("husl", len(genre_counts))
        axes[0].barh(range(len(genre_counts)), genre_counts.values, color=colors, edgecolor='black')
        axes[0].set_yticks(range(len(genre_counts)))
        axes[0].set_yticklabels(genre_counts.index, fontsize=10)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Number of Books', fontsize=12)
        axes[0].set_title(f'Top {top_n} Genres by Book Count', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(genre_counts.values):
            axes[0].text(v + 5, i, str(v), va='center', fontweight='bold')
        
        # Pie chart
        axes[1].pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors)
        axes[1].set_title(f'Top {top_n} Genres Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path / 'genre_distribution.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Genre analysis plot saved to: {output_file}")
        
        plt.show()
        
        return genre_counts
    
    
    def analyze_genre_ratings(self, top_n: int = 10, save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze average ratings by genre
        
        Args:
            top_n: Number of top genres to analyze
            save_plot: Whether to save the plot
            
        Returns:
            DataFrame with genre rating statistics
        """
        logger.info("Analyzing genre ratings...")
        
        if 'genre' not in self.df.columns or 'rating' not in self.df.columns:
            logger.warning("Required columns not found")
            return pd.DataFrame()
        
        # Get top genres by count
        top_genres = self.df['genre'].value_counts().head(top_n).index
        
        # Calculate statistics
        genre_stats = self.df[self.df['genre'].isin(top_genres)].groupby('genre').agg({
            'rating': ['mean', 'median', 'std', 'count']
        }).round(2)
        genre_stats.columns = ['avg_rating', 'median_rating', 'std_rating', 'book_count']
        genre_stats = genre_stats.sort_values('avg_rating', ascending=False).reset_index()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart with error bars
        x = range(len(genre_stats))
        axes[0].bar(x, genre_stats['avg_rating'], yerr=genre_stats['std_rating'], 
                   alpha=0.7, capsize=5, color='steelblue', edgecolor='black')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(genre_stats['genre'], rotation=45, ha='right')
        axes[0].set_ylabel('Average Rating', fontsize=12)
        axes[0].set_title(f'Average Rating by Top {top_n} Genres (with Std Dev)', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim(0, 5.5)
        
        # Add count labels
        for i, row in enumerate(genre_stats.itertuples()):
            axes[0].text(i, row.avg_rating + row.std_rating + 0.1, 
                        f"n={row.book_count}", ha='center', fontsize=8)
        
        # Box plot by genre
        data_for_box = []
        labels_for_box = []
        for genre in genre_stats['genre']:
            genre_data = self.df[self.df['genre'] == genre]['rating'].dropna()
            data_for_box.append(genre_data)
            labels_for_box.append(genre[:20])  # Truncate long names
        
        bp = axes[1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        axes[1].set_xticklabels(labels_for_box, rotation=45, ha='right')
        axes[1].set_ylabel('Rating', fontsize=12)
        axes[1].set_title(f'Rating Distribution by Genre', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path / 'genre_ratings.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Genre ratings plot saved to: {output_file}")
        
        plt.show()
        
        return genre_stats
    
    
    def analyze_authors(self, top_n: int = 15, save_plot: bool = True) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Analyze author statistics
        
        Args:
            top_n: Number of top authors to analyze
            save_plot: Whether to save the plot
            
        Returns:
            Tuple of (author counts, author ratings)
        """
        logger.info("Analyzing authors...")
        
        if 'author' not in self.df.columns:
            logger.warning("Author column not found")
            return pd.Series(), pd.DataFrame()
        
        # Most prolific authors
        author_counts = self.df['author'].value_counts().head(top_n)
        
        # Author ratings (minimum 3 books)
        author_ratings = None
        if 'rating' in self.df.columns:
            author_ratings = self.df.groupby('author').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            author_ratings.columns = ['author', 'avg_rating', 'book_count']
            author_ratings = author_ratings[author_ratings['book_count'] >= 3].sort_values('avg_rating', ascending=False).head(top_n)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Most prolific authors
        colors1 = sns.color_palette("viridis", len(author_counts))
        axes[0].barh(range(len(author_counts)), author_counts.values, color=colors1, edgecolor='black')
        axes[0].set_yticks(range(len(author_counts)))
        axes[0].set_yticklabels(author_counts.index, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Number of Books', fontsize=12)
        axes[0].set_title(f'Top {top_n} Most Prolific Authors', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(author_counts.values):
            axes[0].text(v + 0.5, i, str(v), va='center', fontweight='bold')
        
        # Highest rated authors
        if author_ratings is not None and not author_ratings.empty:
            colors2 = sns.color_palette("rocket", len(author_ratings))
            axes[1].barh(range(len(author_ratings)), author_ratings['avg_rating'], color=colors2, edgecolor='black')
            axes[1].set_yticks(range(len(author_ratings)))
            axes[1].set_yticklabels(author_ratings['author'], fontsize=9)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Average Rating', fontsize=12)
            axes[1].set_title(f'Top {top_n} Highest Rated Authors (min 3 books)', fontsize=14, fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)
            axes[1].set_xlim(0, 5.5)
            
            # Add rating and count labels
            for i, row in enumerate(author_ratings.itertuples()):
                axes[1].text(row.avg_rating + 0.05, i, f"{row.avg_rating:.2f} ({row.book_count})", 
                           va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path / 'author_analysis.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Author analysis plot saved to: {output_file}")
        
        plt.show()
        
        return author_counts, author_ratings
    
    
    def analyze_price(self, save_plot: bool = True) -> Dict:
        """
        Analyze price distribution and statistics
        
        Args:
            save_plot: Whether to save the plot
            
        Returns:
            Dictionary with price statistics
        """
        logger.info("Analyzing prices...")
        
        if 'price' not in self.df.columns:
            logger.warning("Price column not found")
            return {}
        
        stats = {
            'mean': self.df['price'].mean(),
            'median': self.df['price'].median(),
            'std': self.df['price'].std(),
            'min': self.df['price'].min(),
            'max': self.df['price'].max(),
            'q1': self.df['price'].quantile(0.25),
            'q3': self.df['price'].quantile(0.75)
        }
        
        # Remove outliers for visualization
        q1 = stats['q1']
        q3 = stats['q3']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        price_clean = self.df['price'][(self.df['price'] >= lower_bound) & (self.df['price'] <= upper_bound)]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(price_clean, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: ${stats['mean']:.2f}")
        axes[0, 0].axvline(stats['median'], color='blue', linestyle='--', linewidth=2, label=f"Median: ${stats['median']:.2f}")
        axes[0, 0].set_xlabel('Price ($)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Price Distribution (Outliers Removed)', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(self.df['price'].dropna(), vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[0, 1].set_ylabel('Price ($)', fontsize=12)
        axes[0, 1].set_title('Price Box Plot (All Data)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Price ranges
        price_ranges = pd.cut(self.df['price'], bins=[0, 10, 20, 30, 100], labels=['$0-10', '$10-20', '$20-30', '$30+'])
        range_counts = price_ranges.value_counts().sort_index()
        
        axes[1, 0].bar(range(len(range_counts)), range_counts.values, 
                      color=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_xticks(range(len(range_counts)))
        axes[1, 0].set_xticklabels(range_counts.index)
        axes[1, 0].set_ylabel('Number of Books', fontsize=12)
        axes[1, 0].set_title('Books by Price Range', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(range_counts.values):
            axes[1, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # Cumulative distribution
        sorted_prices = np.sort(self.df['price'].dropna())
        cumulative = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices) * 100
        axes[1, 1].plot(sorted_prices, cumulative, linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Price ($)', fontsize=12)
        axes[1, 1].set_ylabel('Cumulative Percentage', fontsize=12)
        axes[1, 1].set_title('Cumulative Price Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path / 'price_analysis.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Price analysis plot saved to: {output_file}")
        
        plt.show()
        
        return stats
    
    
    def analyze_correlations(self, save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze correlations between numeric features
        
        Args:
            save_plot: Whether to save the plot
            
        Returns:
            Correlation matrix
        """
        logger.info("Analyzing correlations...")
        
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
        
        plt.title('Correlation Matrix - Numeric Features', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path / 'correlation_matrix.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to: {output_file}")
        
        plt.show()
        
        return corr_matrix
    
    
    def analyze_price_rating_relationship(self, save_plot: bool = True) -> pd.DataFrame:
        """
        Analyze relationship between price and rating
        
        Args:
            save_plot: Whether to save the plot
            
        Returns:
            DataFrame with price range statistics
        """
        logger.info("Analyzing price-rating relationship...")
        
        if 'price' not in self.df.columns or 'rating' not in self.df.columns:
            logger.warning("Required columns not found")
            return pd.DataFrame()
        
        # Create price bins
        self.df['price_range'] = pd.cut(self.df['price'], 
                                        bins=[0, 10, 20, 30, 100], 
                                        labels=['$0-10', '$10-20', '$20-30', '$30+'])
        
        # Calculate statistics
        price_stats = self.df.groupby('price_range').agg({
            'rating': ['mean', 'median', 'std', 'count']
        }).round(2)
        price_stats.columns = ['avg_rating', 'median_rating', 'std_rating', 'book_count']
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart with error bars
        x = range(len(price_stats))
        axes[0].bar(x, price_stats['avg_rating'], yerr=price_stats['std_rating'],
                   alpha=0.7, capsize=5, color='teal', edgecolor='black')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(price_stats.index)
        axes[0].set_ylabel('Average Rating', fontsize=12)
        axes[0].set_title('Average Rating by Price Range', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim(0, 5.5)
        
        # Add count labels
        for i, row in enumerate(price_stats.itertuples()):
            axes[0].text(i, row.avg_rating + row.std_rating + 0.1, 
                        f"n={row.book_count}", ha='center', fontsize=9)
        
        # Scatter plot
        sample_size = min(1000, len(self.df))
        sample_df = self.df.sample(sample_size)
        
        axes[1].scatter(sample_df['price'], sample_df['rating'], alpha=0.4, s=30, color='navy')
        
        # Add trend line
        z = np.polyfit(sample_df['price'].dropna(), sample_df['rating'].dropna(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(sample_df['price'].min(), sample_df['price'].max(), 100)
        axes[1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        
        axes[1].set_xlabel('Price ($)', fontsize=12)
        axes[1].set_ylabel('Rating', fontsize=12)
        axes[1].set_title(f'Price vs Rating Scatter Plot (n={sample_size})', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path / 'price_rating_relationship.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Price-rating relationship plot saved to: {output_file}")
        
        plt.show()
        
        # Calculate correlation
        corr = self.df[['price', 'rating']].corr().iloc[0, 1]
        logger.info(f"Price-Rating Correlation: {corr:.3f}")
        
        return price_stats
    
    
    def create_interactive_visualizations(self):
        """Create interactive Plotly visualizations"""
        logger.info("Creating interactive visualizations...")
        
        # 1. Interactive scatter plot
        if 'rating' in self.df.columns and 'price' in self.df.columns and 'genre' in self.df.columns:
            sample_df = self.df.sample(min(1000, len(self.df)))
            
            fig1 = px.scatter(
                sample_df,
                x='price',
                y='rating',
                color='genre',
                size='price',
                hover_data=['book_name', 'author'],
                title='Interactive: Price vs Rating by Genre',
                labels={'price': 'Price ($)', 'rating': 'Rating'}
            )
            fig1.show()
        
        # 2. Interactive bar chart - Top genres
        if 'genre' in self.df.columns:
            genre_counts = self.df['genre'].value_counts().head(10)
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=genre_counts.values,
                    y=genre_counts.index,
                    orientation='h',
                    marker=dict(
                        color=genre_counts.values,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=genre_counts.values,
                    textposition='outside'
                )
            ])
            
            fig2.update_layout(
                title='Top 10 Genres (Interactive)',
                xaxis_title='Number of Books',
                yaxis_title='Genre',
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            fig2.show()
        
        # 3. Interactive sunburst - Genre hierarchy
        if 'genre' in self.df.columns and 'rating' in self.df.columns:
            # Create rating categories
            self.df['rating_category'] = pd.cut(self.df['rating'], 
                                                bins=[0, 3, 4, 5], 
                                                labels=['Low', 'Medium', 'High'])
            
            # Sample for performance
            sample_df = self.df.sample(min(500, len(self.df)))
            
            fig3 = px.sunburst(
                sample_df,
                path=['rating_category', 'genre'],
                title='Books by Rating Category and Genre',
                height=600
            )
            fig3.show()
    
    
    def generate_summary_report(self, output_file: str = "eda_summary_report.txt"):
        """
        Generate comprehensive text summary report
        
        Args:
            output_file: Filename for the report
        """
        logger.info("Generating summary report...")
        
        report_path = self.output_path.parent / 'reports'
        report_path.mkdir(parents=True, exist_ok=True)
        
        filepath = report_path / output_file
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS - SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Overview
            overview = self.generate_overview_report()
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 70 + "\n")
            for key, value in overview.items():
                f.write(f"{key.replace('_', ' ').title():30}: {value}\n")
            f.write("\n")
            
            # Rating Analysis
            if 'rating' in self.df.columns:
                f.write("2. RATING ANALYSIS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Mean Rating:              {self.df['rating'].mean():.2f}\n")
                f.write(f"Median Rating:            {self.df['rating'].median():.2f}\n")
                f.write(f"Standard Deviation:       {self.df['rating'].std():.2f}\n")
                f.write(f"Rating Range:             {self.df['rating'].min():.2f} - {self.df['rating'].max():.2f}\n")
                f.write("\n")
            
            # Genre Analysis
            if 'genre' in self.df.columns:
                f.write("3. TOP 10 GENRES\n")
                f.write("-" * 70 + "\n")
                top_genres = self.df['genre'].value_counts().head(10)
                for i, (genre, count) in enumerate(top_genres.items(), 1):
                    percentage = (count / len(self.df)) * 100
                    f.write(f"{i:2}. {genre:40} {count:6} ({percentage:5.1f}%)\n")
                f.write("\n")
            
            # Author Analysis
            if 'author' in self.df.columns:
                f.write("4. TOP 10 AUTHORS (by book count)\n")
                f.write("-" * 70 + "\n")
                top_authors = self.df['author'].value_counts().head(10)
                for i, (author, count) in enumerate(top_authors.items(), 1):
                    f.write(f"{i:2}. {author:40} {count:6} books\n")
                f.write("\n")
            
            # Price Analysis
            if 'price' in self.df.columns:
                f.write("5. PRICE ANALYSIS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Mean Price:               ${self.df['price'].mean():.2f}\n")
                f.write(f"Median Price:             ${self.df['price'].median():.2f}\n")
                f.write(f"Price Range:              ${self.df['price'].min():.2f} - ${self.df['price'].max():.2f}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        logger.info(f"Summary report saved to: {filepath}")
        print(f"\n✅ Summary report saved to: {filepath}")


def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60 + "\n")
    
    # Initialize analyzer
    analyzer = EDAAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Print overview
    analyzer.print_overview()
    
    # Run analyses
    print("\n📊 Analyzing Ratings...")
    analyzer.analyze_ratings()
    
    print("\n📚 Analyzing Genres...")
    analyzer.analyze_genres()
    
    print("\n📈 Analyzing Genre Ratings...")
    analyzer.analyze_genre_ratings()
    
    print("\n✍️  Analyzing Authors...")
    analyzer.analyze_authors()
    
    print("\n💰 Analyzing Prices...")
    analyzer.analyze_price()
    
    print("\n🔗 Analyzing Correlations...")
    analyzer.analyze_correlations()
    
    print("\n💵 Analyzing Price-Rating Relationship...")
    analyzer.analyze_price_rating_relationship()
    
    print("\n🌐 Creating Interactive Visualizations...")
    analyzer.create_interactive_visualizations()
    
    print("\n📄 Generating Summary Report...")
    analyzer.generate_summary_report()
    
    print("\n" + "="*60)
    print("✅ EDA COMPLETE!")
    print("="*60)
    print("\nAll visualizations saved to: outputs/visualizations/")
    print("Summary report saved to: outputs/reports/")


if __name__ == "__main__":
    main()
