"""
Home Page
Welcome page with overview and quick statistics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.components.visualization_panel import VisualizationPanel

# Page config
st.set_page_config(
    page_title="Home - Book Recommender",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# Cache data loading
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
    """Main home page function"""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Book Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #555; margin-bottom: 30px;">
        Discover your next favorite book with AI-powered recommendations
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("⚠️ No data available. Please ensure the dataset is loaded properly.")
        return
    
    # Key Metrics Section
    st.markdown("## 📊 Platform Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size: 2.5rem;">📚</div>
            <div style="font-size: 2rem; font-weight: bold;">{len(df):,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Total Books</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'author' in df.columns:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 2.5rem;">✍️</div>
                <div style="font-size: 2rem; font-weight: bold;">{df['author'].nunique():,}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Authors</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'genre' in df.columns:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 2.5rem;">🎭</div>
                <div style="font-size: 2rem; font-weight: bold;">{df['genre'].nunique():,}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Genres</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'rating' in df.columns:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 2.5rem;">⭐</div>
                <div style="font-size: 2rem; font-weight: bold;">{df['rating'].mean():.2f}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Avg Rating</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        if 'cluster_kmeans' in df.columns:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 2.5rem;">🎯</div>
                <div style="font-size: 2rem; font-weight: bold;">{df['cluster_kmeans'].nunique()}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Book Clusters</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Section
    st.markdown("## 🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Advanced Search</h3>
            <p>Search through thousands of books with powerful filters:</p>
            <ul>
                <li>Filter by genre, author, rating, and price</li>
                <li>Sort and organize results your way</li>
                <li>Export search results in multiple formats</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Multiple AI Models</h3>
            <p>Choose from various recommendation algorithms:</p>
            <ul>
                <li><strong>TF-IDF:</strong> Text-based similarity</li>
                <li><strong>Content-Based:</strong> Feature matching</li>
                <li><strong>Clustering:</strong> Group similar books</li>
                <li><strong>Hybrid:</strong> Best of all methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💡 Smart Recommendations</h3>
            <p>Get personalized book suggestions:</p>
            <ul>
                <li>Based on books you like</li>
                <li>Customized to your preferences</li>
                <li>Discover hidden gems</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Rich Analytics</h3>
            <p>Explore comprehensive data insights:</p>
            <ul>
                <li>Genre trends and distributions</li>
                <li>Author popularity rankings</li>
                <li>Price-rating correlations</li>
                <li>Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats Visualizations
    st.markdown("## 📈 Quick Insights")
    
    viz_panel = VisualizationPanel()
    
    tab1, tab2, tab3 = st.tabs(["📊 Ratings", "🎭 Genres", "💰 Pricing"])
    
    with tab1:
        if 'rating' in df.columns:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                viz_panel.plot_rating_distribution(df, title="Rating Distribution Across All Books")
            
            with col2:
                st.markdown("### 📊 Rating Breakdown")
                
                # Rating categories
                rating_cats = pd.cut(df['rating'], 
                                    bins=[0, 2, 3, 4, 5],
                                    labels=['⭐ Poor (0-2)', '⭐⭐ Fair (2-3)', 
                                           '⭐⭐⭐ Good (3-4)', '⭐⭐⭐⭐ Excellent (4-5)'])
                
                cat_counts = rating_cats.value_counts()
                
                for cat, count in cat_counts.items():
                    percentage = (count / len(df)) * 100
                    st.metric(str(cat), f"{count:,} books", f"{percentage:.1f}%")
    
    with tab2:
        if 'genre' in df.columns:
            viz_panel.plot_genre_distribution(df, top_n=12, chart_type="both")
    
    with tab3:
        if 'price' in df.columns and 'rating' in df.columns:
            viz_panel.plot_price_vs_rating(df, sample_size=1000)
    
    st.markdown("---")
    
    # Popular Books Section
    st.markdown("## 🔥 Popular Books")
    
    if 'rating' in df.columns:
        top_rated = df.nlargest(5, 'rating')
        
        for idx, book in top_rated.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{book['book_name']}**")
                st.caption(f"by {book.get('author', 'Unknown')}")
            
            with col2:
                stars = "⭐" * int(book['rating'])
                st.write(f"{stars} {book['rating']:.1f}")
            
            with col3:
                if 'genre' in book:
                    st.write(f"🎭 {book['genre']}")
    
    st.markdown("---")
    
    # Getting Started Guide
    st.markdown("## 🚀 Getting Started")
    
    st.info("""
    **👈 Use the sidebar** to navigate through different sections:
    
    1. **🔍 Search Books** - Find books using advanced filters
    2. **💡 Recommendations** - Get personalized book suggestions  
    3. **📊 EDA** - Explore detailed data analysis
    4. **📈 Analytics** - View comprehensive statistics and insights
    """)
    
    # Call to Action
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Search Books", use_container_width=True, type="primary"):
            st.switch_page("pages/3_🔍_Search_Books.py")
    
    with col2:
        if st.button("💡 Get Recommendations", use_container_width=True, type="primary"):
            st.switch_page("pages/4_💡_Recommendations.py")
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True, type="primary"):
            st.switch_page("pages/5_📈_Analytics.py")
    
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
