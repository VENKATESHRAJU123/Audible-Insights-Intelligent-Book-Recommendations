"""
Recommendations Page
Get personalized book recommendations using AI models
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recommenders import (
    TFIDFRecommender,
    ContentBasedRecommender,
    ClusteringRecommender,
    HybridRecommender
)
from app.components.recommendation_display import RecommendationDisplay
from app.components.user_preferences import UserPreferences

# Page config
st.set_page_config(
    page_title="Recommendations - Book Recommender",
    page_icon="💡",
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


@st.cache_resource
def load_models(df):
    """Load all recommendation models"""
    models = {}
    
    try:
        with st.spinner("Loading recommendation models..."):
            # TF-IDF Recommender
            tfidf_rec = TFIDFRecommender()
            tfidf_rec.fit(df)
            models['TF-IDF'] = tfidf_rec
            
            # Content-Based Recommender
            content_rec = ContentBasedRecommender()
            content_rec.fit(df)
            models['Content-Based'] = content_rec
            
            # Clustering Recommender
            cluster_rec = ClusteringRecommender(n_clusters=8)
            cluster_rec.fit(df)
            models['Clustering'] = cluster_rec
            
            # Hybrid Recommender
            hybrid_rec = HybridRecommender()
            hybrid_rec.fit(df)
            models['Hybrid'] = hybrid_rec
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}


def main():
    """Main recommendations page function"""
    
    st.title("💡 Book Recommendations")
    st.markdown("### Discover your next favorite book with AI-powered suggestions")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("⚠️ No data available")
        return
    
    # Load models
    models = load_models(df)
    
    if not models:
        st.error("⚠️ Recommendation models not available")
        return
    
    # Initialize components
    display = RecommendationDisplay()
    user_prefs = UserPreferences()
    
    # Tabs for different recommendation methods
    tab1, tab2, tab3 = st.tabs([
        "📖 Based on a Book", 
        "🎯 Based on Preferences", 
        "📚 Reading History"
    ])
    
    # TAB 1: Book-Based Recommendations
    with tab1:
        st.markdown("## 📖 Find Similar Books")
        st.markdown("Select a book you like, and we'll recommend similar ones!")
        
        # Input method selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_method = st.radio(
                "How would you like to find a book?",
                ["🔍 Search by Name", "📋 Select from List", "⭐ Top Rated Books"],
                key="input_method"
            )
        
        with col2:
            # Model selection
            model_choice = st.selectbox(
                "Recommendation Model",
                ["🔥 Hybrid (Best)", "📝 TF-IDF", "🎯 Content-Based", "🎪 Clustering"],
                key="model_choice"
            )
        
        selected_book = None
        
        # Input method: Search
        if input_method == "🔍 Search by Name":
            search_query = st.text_input(
                "Enter book name",
                placeholder="e.g., Harry Potter, The Great Gatsby...",
                key="book_search"
            )
            
            if search_query:
                matching_books = df[
                    df['book_name'].str.contains(search_query, case=False, na=False)
                ]['book_name'].tolist()
                
                if matching_books:
                    selected_book = st.selectbox(
                        "Select a book",
                        matching_books,
                        key="book_select_search"
                    )
                else:
                    st.warning("No books found matching your search")
        
        # Input method: List
        elif input_method == "📋 Select from List":
            # Group by genre for easier selection
            if 'genre' in df.columns:
                selected_genre = st.selectbox(
                    "Filter by genre (optional)",
                    ['All Genres'] + sorted(df['genre'].unique().tolist()),
                    key="genre_filter_list"
                )
                
                if selected_genre == 'All Genres':
                    book_list = df['book_name'].tolist()
                else:
                    book_list = df[df['genre'] == selected_genre]['book_name'].tolist()
            else:
                book_list = df['book_name'].tolist()
            
            selected_book = st.selectbox(
                "Select a book",
                book_list,
                key="book_select_list"
            )
        
        # Input method: Top Rated
        else:
            if 'rating' in df.columns:
                top_rated = df.nlargest(50, 'rating')
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    selected_book = st.selectbox(
                        "Select from top-rated books",
                        top_rated['book_name'].tolist(),
                        key="book_select_top"
                    )
                
                with col2:
                    if selected_book:
                        book_rating = top_rated[top_rated['book_name'] == selected_book]['rating'].values[0]
                        st.metric("Rating", f"{book_rating:.1f}⭐")
        
        # Recommendation settings
        if selected_book:
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_recommendations = st.slider(
                    "Number of recommendations",
                    min_value=5,
                    max_value=20,
                    value=10,
                    key="num_recs"
                )
            
            with col2:
                show_scores = st.checkbox(
                    "Show match scores",
                    value=True,
                    key="show_scores"
                )
            
            with col3:
                show_comparison = st.checkbox(
                    "Show comparison table",
                    value=False,
                    key="show_comparison"
                )
            
            # Get recommendations button
            if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
                
                # Map model choice to model
                model_map = {
                    "🔥 Hybrid (Best)": "Hybrid",
                    "📝 TF-IDF": "TF-IDF",
                    "🎯 Content-Based": "Content-Based",
                    "🎪 Clustering": "Clustering"
                }
                
                model = models[model_map[model_choice]]
                
                # Get recommendations
                with st.spinner("Generating recommendations..."):
                    try:
                        recommendations = model.get_recommendations(
                            selected_book,
                            top_n=num_recommendations
                        )
                        
                        if recommendations.empty:
                            display.display_no_recommendations_message(selected_book)
                        else:
                            st.success(f"✅ Found {len(recommendations)} recommendations!")
                            
                            # Display selected book
                            st.markdown("---")
                            st.markdown("### 📖 You Selected:")
                            selected_book_data = df[df['book_name'] == selected_book].iloc[0]
                            display.display_book_card(selected_book_data)
                            
                            # Add to reading history
                            user_prefs.add_to_reading_history(
                                selected_book,
                                selected_book_data.get('author', 'Unknown'),
                                selected_book_data.get('genre', 'Unknown')
                            )
                            
                            # Display metrics
                            display.display_recommendation_metrics(recommendations)
                            
                            # Comparison table
                            if show_comparison:
                                st.markdown("---")
                                display.display_comparison_table(
                                    selected_book_data,
                                    recommendations,
                                    num_recommendations=min(5, num_recommendations)
                                )
                            
                            # Display recommendations
                            st.markdown("---")
                            st.markdown("### 🎯 Recommended Books:")
                            
                            # View mode
                            view_mode = st.radio(
                                "Display Mode",
                                ["Cards", "Grid", "List"],
                                horizontal=True,
                                key="rec_view_mode"
                            )
                            
                            if view_mode == "Cards":
                                for idx, row in recommendations.iterrows():
                                    score = row.get('similarity_score', row.get('hybrid_score', None))
                                    display.display_book_card(
                                        row, 
                                        show_score=show_scores, 
                                        score=score
                                    )
                            
                            elif view_mode == "Grid":
                                display.display_recommendation_grid(
                                    recommendations,
                                    num_columns=3,
                                    show_scores=show_scores
                                )
                            
                            else:  # List
                                for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                                    score = row.get('similarity_score', row.get('hybrid_score', 0))
                                    
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        st.markdown(f"**{i}. {row['book_name']}**")
                                        st.caption(f"by {row.get('author', 'Unknown')}")
                                    
                                    with col2:
                                        if 'rating' in row:
                                            stars = "⭐" * int(row['rating'])
                                            st.write(f"{stars} {row['rating']:.1f}")
                                    
                                    with col3:
                                        if show_scores:
                                            st.metric("Match", f"{score*100:.0f}%")
                            
                            # Export recommendations
                            st.markdown("---")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                csv = recommendations.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv,
                                    file_name=f"recommendations_{selected_book.replace(' ', '_')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Create shareable link (placeholder)
                                if st.button("🔗 Share Recommendations", use_container_width=True):
                                    st.info("Sharing feature coming soon!")
                            
                            with col3:
                                if st.button("💾 Save to My List", use_container_width=True):
                                    st.success("Saved! (Feature coming soon)")
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
    
    # TAB 2: Preference-Based Recommendations
    with tab2:
        st.markdown("## 🎯 Based on Your Preferences")
        st.markdown("Tell us what you like, and we'll find books for you!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre preferences
            if 'genre' in df.columns:
                st.markdown("### 📚 Preferred Genres")
                selected_genres = st.multiselect(
                    "Select genres you enjoy",
                    sorted(df['genre'].unique().tolist()),
                    key="pref_genres_select"
                )
            else:
                selected_genres = []
            
            # Rating preference
            if 'rating' in df.columns:
                st.markdown("### ⭐ Rating Preference")
                min_rating = st.slider(
                    "Minimum rating",
                    0.0, 5.0, 4.0, 0.5,
                    key="pref_min_rating"
                )
        
        with col2:
            # Author preferences
            if 'author' in df.columns:
                st.markdown("### ✍️ Favorite Authors")
                top_authors = df['author'].value_counts().head(30).index.tolist()
                selected_authors = st.multiselect(
                    "Select authors you like",
                    top_authors,
                    key="pref_authors_select"
                )
            else:
                selected_authors = []
            
            # Price range
            if 'price' in df.columns:
                st.markdown("### 💰 Price Range")
                max_price = st.slider(
                    "Maximum price ($)",
                    0.0, float(df['price'].max()), 30.0, 5.0,
                    key="pref_max_price"
                )
        
        # Number of recommendations
        num_pref_recs = st.slider(
            "Number of recommendations",
            5, 30, 15,
            key="num_pref_recs"
        )
        
        # Get recommendations
        if st.button("🎯 Find Books", type="primary", use_container_width=True):
            with st.spinner("Searching for books..."):
                filtered = df.copy()
                
                # Apply filters
                if selected_genres:
                    filtered = filtered[filtered['genre'].isin(selected_genres)]
                
                if selected_authors:
                    filtered = filtered[filtered['author'].isin(selected_authors)]
                
                if 'rating' in filtered.columns:
                    filtered = filtered[filtered['rating'] >= min_rating]
                
                if 'price' in filtered.columns:
                    filtered = filtered[filtered['price'] <= max_price]
                
                if filtered.empty:
                    st.warning("No books found matching your preferences. Try adjusting your criteria.")
                else:
                    # Sort by rating and get top N
                    recommended = filtered.sort_values('rating', ascending=False).head(num_pref_recs)
                    
                    st.success(f"✅ Found {len(recommended)} books matching your preferences!")
                    
                    # Display metrics
                    display.display_recommendation_metrics(recommended)
                    
                    st.markdown("---")
                    st.markdown("### 📚 Recommended Books:")
                    
                    for idx, row in recommended.iterrows():
                        display.display_book_card(row)
    
    # TAB 3: Reading History
    with tab3:
        st.markdown("## 📚 Based on Your Reading History")
        
        reading_history = user_prefs.get_preference('reading_history', [])
        
        if not reading_history:
            st.info("""
            📖 Your reading history is empty.
            
            When you view book recommendations, they'll be automatically added to your history.
            You can also manually add books in the User Preferences page.
            """)
        else:
            st.markdown(f"**You have {len(reading_history)} books in your reading history**")
            
            # Display history
            with st.expander("📖 View Reading History"):
                for i, book in enumerate(reading_history, 1):
                    st.markdown(f"{i}. **{book['title']}** by {book['author']} - *{book['genre']}*")
            
            # Get recommendations based on history
            if st.button("🎯 Get Recommendations from History", type="primary", use_container_width=True):
                st.info("Feature coming soon! Recommendations based on reading history.")


if __name__ == "__main__":
    main()
