"""
Search Books Page
Advanced book search with filters and export options
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.components.search_interface import SearchInterface
from app.components.recommendation_display import RecommendationDisplay
from app.components.user_preferences import UserPreferences

# Page config
st.set_page_config(
    page_title="Search Books - Book Recommender",
    page_icon="🔍",
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
    """Main search page function"""
    
    st.title("🔍 Search & Discover Books")
    st.markdown("### Find your next great read with advanced search filters")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("⚠️ No data available")
        return
    
    # Initialize components
    search_ui = SearchInterface()
    display = RecommendationDisplay()
    user_prefs = UserPreferences()
    
    # Option to apply user preferences
    use_preferences = st.checkbox(
        "🎯 Apply my saved preferences",
        help="Filter results based on your saved preferences"
    )
    
    if use_preferences:
        df = user_prefs.apply_preferences_to_dataframe(df)
        st.info(f"✅ User preferences applied. Showing {len(df)} books matching your preferences.")
    
    # Search bar
    st.markdown("---")
    search_query = search_ui.render_search_bar(df)
    
    # Quick search suggestions
    if not search_query:
        st.markdown("#### 💡 Quick Search Suggestions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Popular Genres**")
            if 'genre' in df.columns:
                top_genres = df['genre'].value_counts().head(5).index.tolist()
                for genre in top_genres:
                    st.caption(f"• {genre}")
        
        with col2:
            st.markdown("**Top Rated Books**")
            if 'rating' in df.columns and 'book_name' in df.columns:
                top_rated = df.nlargest(5, 'rating')['book_name'].tolist()
                for book in top_rated:
                    st.caption(f"• {book[:40]}...")
        
        with col3:
            st.markdown("**Popular Authors**")
            if 'author' in df.columns:
                top_authors = df['author'].value_counts().head(5).index.tolist()
                for author in top_authors:
                    st.caption(f"• {author}")
    
    # Filters in sidebar
    filters = search_ui.render_filter_sidebar(df)
    
    # Apply search and filters
    filtered_df = search_ui.apply_filters(df, filters, search_query)
    
    st.markdown("---")
    
    # Display filter summary
    search_ui.render_filter_summary(filters, len(filtered_df))
    
    if filtered_df.empty:
        st.warning("""
        😕 No books found matching your criteria. 
        
        **Try:**
        - Removing some filters
        - Broadening your search query
        - Checking spelling
        """)
        return
    
    # Sort options
    sort_column, ascending, items_per_page = search_ui.render_sort_options(filtered_df)
    
    # Apply sorting
    sorted_df = filtered_df.sort_values(sort_column, ascending=ascending)
    
    # Pagination
    start_idx, end_idx = search_ui.render_pagination(len(sorted_df), items_per_page)
    
    # Display options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 📚 Search Results")
    
    with col2:
        view_mode = st.selectbox(
            "View Mode",
            ["Cards", "Compact", "Detailed", "Table"],
            key="view_mode"
        )
    
    with col3:
        show_descriptions = st.checkbox(
            "Show Descriptions",
            value=False,
            key="show_desc"
        )
    
    st.markdown("---")
    
    # Display results
    page_df = sorted_df.iloc[start_idx:end_idx]
    
    if view_mode == "Cards":
        for idx, row in page_df.iterrows():
            display.display_book_card(
                row, 
                show_description=show_descriptions,
                card_style="default"
            )
    
    elif view_mode == "Compact":
        for idx, row in page_df.iterrows():
            display.display_book_card(row, card_style="compact")
    
    elif view_mode == "Detailed":
        for idx, row in page_df.iterrows():
            display.display_book_card(row, card_style="detailed")
    
    else:  # Table
        display_cols = ['book_name', 'author', 'genre', 'rating']
        
        if 'price' in page_df.columns:
            display_cols.append('price')
        
        review_cols = [col for col in page_df.columns if 'review' in col.lower()]
        if review_cols:
            display_cols.append(review_cols[0])
        
        # Format the dataframe
        styled_df = page_df[display_cols].copy()
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "book_name": st.column_config.TextColumn("Book Name", width="medium"),
                "author": st.column_config.TextColumn("Author", width="medium"),
                "genre": st.column_config.TextColumn("Genre", width="small"),
                "rating": st.column_config.NumberColumn("Rating", format="⭐ %.1f"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f") if 'price' in display_cols else None
            }
        )
    
    # Quick Stats
    st.markdown("---")
    st.markdown("### 📊 Results Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Results", f"{len(sorted_df):,}")
    
    with col2:
        if 'rating' in sorted_df.columns:
            st.metric("Avg Rating", f"{sorted_df['rating'].mean():.2f}⭐")
    
    with col3:
        if 'genre' in sorted_df.columns:
            st.metric("Unique Genres", sorted_df['genre'].nunique())
    
    with col4:
        if 'author' in sorted_df.columns:
            st.metric("Unique Authors", sorted_df['author'].nunique())
    
    # Export options
    st.markdown("---")
    search_ui.render_export_options(sorted_df, "search_results")
    
    # Save search
    st.markdown("---")
    if st.button("💾 Save This Search"):
        st.success("Search saved! (Feature coming soon)")


if __name__ == "__main__":
    main()
