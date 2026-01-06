"""
Book Recommendation System - Streamlit Application
A comprehensive web interface for book recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    .book-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


# Cache data loading
@st.cache_data
def load_data():
    """Load the book dataset"""
    try:
        df = pd.read_csv('data/processed/clustered_data.csv')
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def main():
    """Main application function"""
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Book Recommender")
        st.markdown("---")
        
        # Navigation info
        st.markdown("### 📍 Navigation")
        st.info("Use the sidebar pages above to navigate through different sections")
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        df = load_data()
        
        if not df.empty:
            st.metric("Total Books", f"{len(df):,}")
            
            # Find rating column
            rating_col = None
            for col in ['rating', 'Rating']:
                if col in df.columns:
                    rating_col = col
                    break
            
            if rating_col:
                st.metric("Avg Rating", f"{df[rating_col].mean():.2f}⭐")
            
            # Find genre column
            genre_col = None
            for col in ['genre', 'Genre']:
                if col in df.columns:
                    genre_col = col
                    break
            
            if genre_col:
                st.metric("Genres", f"{df[genre_col].nunique()}")
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.markdown("""
        AI-powered book recommendation system using:
        - TF-IDF Analysis
        - Content-Based Filtering
        - K-Means Clustering
        - Hybrid Models
        """)
    
    # Main content
    st.title("📚 Book Recommendation System")
    st.markdown("### Discover Your Next Favorite Book!")
    
    # Introduction
    st.markdown("""
    Welcome to our intelligent Book Recommendation System! This application uses advanced 
    machine learning algorithms to help you discover books tailored to your preferences.
    """)
    
    # Load data for statistics
    df = load_data()
    
    if not df.empty:
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Get column names (handle case variations)
        book_col = None
        for col in ['book_name', 'Book Name', 'book name']:
            if col in df.columns:
                book_col = col
                break
        
        author_col = None
        for col in ['author', 'Author']:
            if col in df.columns:
                author_col = col
                break
        
        genre_col = None
        for col in ['genre', 'Genre']:
            if col in df.columns:
                genre_col = col
                break
        
        rating_col = None
        for col in ['rating', 'Rating']:
            if col in df.columns:
                rating_col = col
                break
        
        with col1:
            st.metric("📚 Total Books", f"{len(df):,}")
        
        with col2:
            if author_col:
                st.metric("✍️ Authors", f"{df[author_col].nunique():,}")
        
        with col3:
            if genre_col:
                st.metric("🎭 Genres", f"{df[genre_col].nunique():,}")
        
        with col4:
            if rating_col:
                st.metric("⭐ Avg Rating", f"{df[rating_col].mean():.2f}")
        
        st.markdown("---")
        
        # Features section
        st.markdown("## 🎯 Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Multiple Recommendation Engines
            - **TF-IDF Based**: Content similarity using text analysis
            - **Content-Based**: Feature-based filtering
            - **Clustering**: Group similar books together
            - **Hybrid**: Combines multiple approaches
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Rich Analytics
            - Explore genre distributions
            - Analyze rating patterns
            - Discover popular authors
            - Interactive visualizations
            """)
        
        st.markdown("---")
        
        # Getting started
        st.markdown("## 🚀 Getting Started")
        st.info("👈 Use the **sidebar pages** (Home, EDA, Search Books, etc.) to navigate through different sections!")
        
        st.markdown("---")
        
        # Quick access buttons
        st.markdown("## 🔗 Quick Access")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔍 Search Books")
            st.markdown("Find books using advanced filters")
            if st.button("Go to Search", key="btn_search", use_container_width=True):
                st.info("Click on '🔍 Search Books' in the sidebar pages")
        
        with col2:
            st.markdown("### 💡 Get Recommendations")
            st.markdown("Discover similar books")
            if st.button("Go to Recommendations", key="btn_recs", use_container_width=True):
                st.info("Click on '💡 Recommendations' in the sidebar pages")
        
        with col3:
            st.markdown("### 📊 View Analytics")
            st.markdown("Explore data insights")
            if st.button("Go to Analytics", key="btn_analytics", use_container_width=True):
                st.info("Click on '📈 Analytics' in the sidebar pages")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>📚 Book Recommendation System | Built with Streamlit & Machine Learning</p>
        <p style="font-size: 0.9rem;">Powered by TF-IDF, Content-Based Filtering, and K-Means Clustering</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
