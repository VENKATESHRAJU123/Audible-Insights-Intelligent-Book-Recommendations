"""
Analysis Questions Page
Answers to the 5 key analysis questions with interactive visualizations
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

# Page config
st.set_page_config(
    page_title="Analysis Questions",
    page_icon="❓",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load the book dataset"""
    try:
        df = pd.read_csv('data/processed/clustered_data.csv')
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def question1_popular_genres(df):
    """Question 1: What are the most popular genres?"""
    
    st.markdown("## 📚 Question 1: Most Popular Genres")
    
    if 'genre' not in df.columns:
        st.warning("Genre data not available")
        return
    
    genre_counts = df['genre'].value_counts()
    top_genres = genre_counts.head(15)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Genres", df['genre'].nunique())
    
    with col2:
        st.metric("Most Popular", top_genres.index[0])
    
    with col3:
        st.metric("Books in Top Genre", top_genres.values[0])
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart", "📋 Table"])
    
    with tab1:
        # Interactive bar chart
        fig = px.bar(
            x=top_genres.values,
            y=top_genres.index,
            orientation='h',
            title="Top 15 Most Popular Genres",
            labels={'x': 'Number of Books', 'y': 'Genre'},
            color=top_genres.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Pie chart
        fig = px.pie(
            values=top_genres.values,
            names=top_genres.index,
            title="Genre Distribution (Top 15)"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Data table
        genre_df = pd.DataFrame({
            'Genre': top_genres.index,
            'Book Count': top_genres.values,
            'Percentage': (top_genres.values / len(df) * 100).round(2)
        })
        st.dataframe(genre_df, use_container_width=True, hide_index=True)
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    st.info(f"""
    - **Most Popular Genre**: {top_genres.index[0]} ({top_genres.values[0]} books, {top_genres.values[0]/len(df)*100:.1f}%)
    - **Top 3 Genres** account for **{top_genres.head(3).sum()/len(df)*100:.1f}%** of all books
    - **Total Unique Genres**: {df['genre'].nunique()}
    """)


def question2_highest_rated_authors(df):
    """Question 2: Which authors have the highest-rated books?"""
    
    st.markdown("## ✍️ Question 2: Highest-Rated Authors")
    
    if 'author' not in df.columns or 'rating' not in df.columns:
        st.warning("Required data not available")
        return
    
    # Calculate author statistics
    author_stats = df.groupby('author').agg({
        'book_name': 'count',
        'rating': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    author_stats.columns = ['book_count', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
    author_stats = author_stats.reset_index()
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        min_books = st.slider("Minimum books by author", 1, 10, 3)
    
    with col2:
        top_n = st.slider("Number of authors to show", 5, 30, 15)
    
    # Filter and sort
    qualified_authors = author_stats[author_stats['book_count'] >= min_books]
    top_authors = qualified_authors.sort_values('avg_rating', ascending=False).head(top_n)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Authors", df['author'].nunique())
    
    with col2:
        st.metric("Qualified Authors", len(qualified_authors))
    
    with col3:
        if not top_authors.empty:
            st.metric("Highest Avg Rating", f"{top_authors.iloc[0]['avg_rating']:.2f}⭐")
    
    # Visualizations
    tab1, tab2 = st.tabs(["📊 Highest Rated", "📚 Most Prolific"])
    
    with tab1:
        if not top_authors.empty:
            fig = px.bar(
                top_authors,
                x='avg_rating',
                y='author',
                orientation='h',
                title=f"Top {len(top_authors)} Highest Rated Authors (min {min_books} books)",
                labels={'avg_rating': 'Average Rating', 'author': 'Author'},
                color='avg_rating',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.dataframe(
                top_authors[['author', 'avg_rating', 'book_count', 'min_rating', 'max_rating']],
                use_container_width=True,
                hide_index=True
            )
    
    with tab2:
        most_prolific = author_stats.sort_values('book_count', ascending=False).head(15)
        
        fig = px.bar(
            most_prolific,
            x='book_count',
            y='author',
            orientation='h',
            title="Most Prolific Authors",
            labels={'book_count': 'Number of Books', 'author': 'Author'},
            color='avg_rating',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def question3_rating_distribution(df):
    """Question 3: What is the average rating distribution?"""
    
    st.markdown("## ⭐ Question 3: Rating Distribution")
    
    if 'rating' not in df.columns:
        st.warning("Rating data not available")
        return
    
    # Statistics
    stats = df['rating'].describe()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Rating", f"{stats['mean']:.2f}⭐")
    
    with col2:
        st.metric("Median Rating", f"{stats['50%']:.2f}⭐")
    
    with col3:
        st.metric("Std Deviation", f"{stats['std']:.2f}")
    
    with col4:
        st.metric("Range", f"{stats['min']:.1f} - {stats['max']:.1f}")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📦 Box Plot", "📋 Categories"])
    
    with tab1:
        # Histogram
        fig = px.histogram(
            df,
            x='rating',
            nbins=30,
            title="Rating Distribution",
            labels={'rating': 'Rating', 'count': 'Number of Books'}
        )
        
        # Add mean and median lines
        fig.add_vline(x=stats['mean'], line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {stats['mean']:.2f}")
        fig.add_vline(x=stats['50%'], line_dash="dash", line_color="green",
                     annotation_text=f"Median: {stats['50%']:.2f}")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Box plot
        fig = px.box(
            df,
            y='rating',
            title="Rating Box Plot",
            labels={'rating': 'Rating'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Category breakdown
        rating_cats = pd.cut(df['rating'], 
                            bins=[0, 2, 3, 4, 5],
                            labels=['⭐ Poor (0-2)', '⭐⭐ Fair (2-3)', 
                                   '⭐⭐⭐ Good (3-4)', '⭐⭐⭐⭐ Excellent (4-5)'])
        
        cat_counts = rating_cats.value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=cat_counts.index.astype(str),
                y=cat_counts.values,
                title="Books by Rating Category",
                labels={'x': 'Category', 'y': 'Number of Books'},
                color=cat_counts.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show percentage breakdown
            for cat, count in cat_counts.items():
                percentage = (count / len(df)) * 100
                st.metric(str(cat), f"{count} books", f"{percentage:.1f}%")


def question4_publication_trends(df):
    """Question 4: Are there trends in publication years?"""
    
    st.markdown("## 📅 Question 4: Publication Year Trends")
    
    if 'year' not in df.columns:
        st.warning("⚠️ Publication year data not available in dataset")
        st.info("This analysis requires a 'year' or 'publication_year' column in the data.")
        return
    
    # Year statistics
    year_stats = df.groupby('year').agg({
        'book_name': 'count',
        'rating': 'mean'
    }).round(2)
    year_stats.columns = ['book_count', 'avg_rating']
    year_stats = year_stats.sort_index()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Year Range", f"{df['year'].min()} - {df['year'].max()}")
    
    with col2:
        peak_year = year_stats['book_count'].idxmax()
        st.metric("Peak Year", peak_year)
    
    with col3:
        st.metric("Peak Year Books", year_stats.loc[peak_year, 'book_count'])
    
    with col4:
        best_year = year_stats['avg_rating'].idxmax()
        st.metric("Highest Rated Year", f"{best_year} ({year_stats.loc[best_year, 'avg_rating']:.2f}⭐)")
    
    # Visualizations
    tab1, tab2 = st.tabs(["📊 Books by Year", "⭐ Ratings by Year"])
    
    with tab1:
        fig = px.bar(
            year_stats.reset_index(),
            x='year',
            y='book_count',
            title="Books Published by Year",
            labels={'year': 'Year', 'book_count': 'Number of Books'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(
            year_stats.reset_index(),
            x='year',
            y='avg_rating',
            title="Average Rating by Publication Year",
            labels={'year': 'Year', 'avg_rating': 'Average Rating'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)


def question5_ratings_vs_reviews(df):
    """Question 5: How do ratings vary with review counts?"""
    
    st.markdown("## 📊 Question 5: Ratings vs Review Counts")
    
    if 'rating' not in df.columns:
        st.warning("Rating data not available")
        return
    
    if 'number_of_reviews' not in df.columns:
        st.warning("Review count data not available")
        return
    
    # Create categories
    df['review_category'] = pd.cut(df['number_of_reviews'], 
                                   bins=[0, 50, 200, 500, 1000, 10000],
                                   labels=['Very Low (0-50)', 'Low (50-200)', 
                                          'Medium (200-500)', 'High (500-1k)', 
                                          'Very High (1k+)'])
    
    # Statistics by category
    cat_stats = df.groupby('review_category').agg({
        'rating': ['mean', 'count'],
        'number_of_reviews': 'mean'
    }).round(2)
    cat_stats.columns = ['avg_rating', 'book_count', 'avg_reviews']
    
    # Correlation
    correlation = df[['rating', 'number_of_reviews']].corr().iloc[0, 1]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Correlation", f"{correlation:.4f}")
    
    with col2:
        highest_rated_cat = cat_stats['avg_rating'].idxmax()
        st.metric("Highest Rated Category", str(highest_rated_cat))
    
    with col3:
        st.metric("Avg Rating", f"{cat_stats.loc[highest_rated_cat, 'avg_rating']:.2f}⭐")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 By Category", "🔍 Scatter Plot", "📋 Statistics"])
    
    with tab1:
        # Bar chart by category
        fig = px.bar(
            cat_stats.reset_index(),
            x='review_category',
            y='avg_rating',
            title="Average Rating by Review Count Category",
            labels={'review_category': 'Review Count Category', 'avg_rating': 'Average Rating'},
            color='avg_rating',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Scatter plot
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size)
        
        fig = px.scatter(
            sample_df,
            x='number_of_reviews',
            y='rating',
            title=f"Rating vs Review Count (Sample: {sample_size} books)",
            labels={'number_of_reviews': 'Number of Reviews', 'rating': 'Rating'},
            trendline="ols",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation interpretation
        if abs(correlation) < 0.3:
            strength = "weak"
        elif abs(correlation) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        st.info(f"📊 The correlation of **{correlation:.4f}** indicates a **{strength}** relationship between ratings and review counts.")
    
    with tab3:
        # Statistics table
        st.dataframe(cat_stats.reset_index(), use_container_width=True, hide_index=True)


def main():
    """Main function"""
    
    st.title("❓ Analysis Questions")
    st.markdown("### Comprehensive Answers to Key Business Questions")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("⚠️ No data available")
        return
    
    # Sidebar - Question selector
    with st.sidebar:
        st.markdown("## 📑 Select Question")
        
        question = st.radio(
            "Choose a question to explore:",
            [
                "1️⃣ Most Popular Genres",
                "2️⃣ Highest-Rated Authors",
                "3️⃣ Rating Distribution",
                "4️⃣ Publication Trends",
                "5️⃣ Ratings vs Reviews",
                "📊 View All Questions"
            ]
        )
        
        st.markdown("---")
        st.info("💡 Each question includes interactive visualizations and insights")
    
    # Display selected question
    if question == "📊 View All Questions":
        question1_popular_genres(df)
        st.markdown("---")
        question2_highest_rated_authors(df)
        st.markdown("---")
        question3_rating_distribution(df)
        st.markdown("---")
        question4_publication_trends(df)
        st.markdown("---")
        question5_ratings_vs_reviews(df)
    
    elif question == "1️⃣ Most Popular Genres":
        question1_popular_genres(df)
    
    elif question == "2️⃣ Highest-Rated Authors":
        question2_highest_rated_authors(df)
    
    elif question == "3️⃣ Rating Distribution":
        question3_rating_distribution(df)
    
    elif question == "4️⃣ Publication Trends":
        question4_publication_trends(df)
    
    elif question == "5️⃣ Ratings vs Reviews":
        question5_ratings_vs_reviews(df)


if __name__ == "__main__":
    main()
