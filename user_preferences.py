"""
User Preferences Component
Manages user preferences and personalization settings
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class UserPreferences:
    """Component for managing user preferences and settings"""
    
    def __init__(self):
        """Initialize UserPreferences"""
        self.preferences_file = Path("data/user_preferences.json")
        self.session_key = "user_preferences"
        
        # Initialize session state
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = self.load_preferences()
    
    
    def load_preferences(self) -> Dict:
        """
        Load user preferences from file
        
        Returns:
            Dictionary with user preferences
        """
        default_prefs = {
            'favorite_genres': [],
            'favorite_authors': [],
            'reading_history': [],
            'rating_preference': (0.0, 5.0),
            'price_range': (0.0, 100.0),
            'items_per_page': 10,
            'default_sort': 'Rating (High to Low)',
            'recommendation_model': 'Hybrid',
            'theme': 'Light',
            'show_descriptions': False,
            'min_reviews': 0,
            'exclude_genres': [],
            'language_preference': 'English'
        }
        
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    loaded_prefs = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_prefs.update(loaded_prefs)
        except Exception as e:
            st.warning(f"Could not load preferences: {e}")
        
        return default_prefs
    
    
    def save_preferences(self, preferences: Dict):
        """
        Save user preferences to file
        
        Args:
            preferences: Dictionary with user preferences
        """
        try:
            # Create directory if it doesn't exist
            self.preferences_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.preferences_file, 'w') as f:
                json.dump(preferences, f, indent=2)
            
            st.session_state[self.session_key] = preferences
            st.success("✅ Preferences saved successfully!")
        except Exception as e:
            st.error(f"Error saving preferences: {e}")
    
    
    def get_preference(self, key: str, default=None):
        """
        Get a specific preference value
        
        Args:
            key: Preference key
            default: Default value if key not found
            
        Returns:
            Preference value
        """
        return st.session_state[self.session_key].get(key, default)
    
    
    def set_preference(self, key: str, value):
        """
        Set a specific preference value
        
        Args:
            key: Preference key
            value: Preference value
        """
        st.session_state[self.session_key][key] = value
    
    
    def render_preferences_panel(self, df: pd.DataFrame):
        """
        Render complete preferences management panel
        
        Args:
            df: DataFrame for generating preference options
        """
        st.markdown("## ⚙️ User Preferences")
        st.markdown("Customize your experience and save your preferences")
        
        prefs = st.session_state[self.session_key]
        
        # Create tabs for different preference categories
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Content Preferences", 
            "🎨 Display Settings", 
            "🤖 Recommendation Settings",
            "📚 Reading History"
        ])
        
        # TAB 1: Content Preferences
        with tab1:
            st.markdown("### 📖 Content Preferences")
            
            # Favorite Genres
            if 'genre' in df.columns:
                st.markdown("#### Favorite Genres")
                all_genres = sorted(df['genre'].dropna().unique().tolist())
                
                favorite_genres = st.multiselect(
                    "Select your favorite genres",
                    all_genres,
                    default=prefs.get('favorite_genres', []),
                    key="pref_genres"
                )
                prefs['favorite_genres'] = favorite_genres
                
                # Exclude Genres
                st.markdown("#### Genres to Exclude")
                exclude_genres = st.multiselect(
                    "Select genres to exclude from recommendations",
                    all_genres,
                    default=prefs.get('exclude_genres', []),
                    key="pref_exclude_genres"
                )
                prefs['exclude_genres'] = exclude_genres
            
            st.markdown("---")
            
            # Favorite Authors
            if 'author' in df.columns:
                st.markdown("#### Favorite Authors")
                
                # Option to search or select from top authors
                author_input_method = st.radio(
                    "How would you like to add authors?",
                    ["Select from Top Authors", "Search for Authors"],
                    key="author_input_method"
                )
                
                if author_input_method == "Select from Top Authors":
                    top_authors = df['author'].value_counts().head(50).index.tolist()
                    favorite_authors = st.multiselect(
                        "Select favorite authors",
                        top_authors,
                        default=[a for a in prefs.get('favorite_authors', []) if a in top_authors],
                        key="pref_authors_top"
                    )
                else:
                    author_search = st.text_input(
                        "Search for authors",
                        key="author_search"
                    )
                    
                    if author_search:
                        matching_authors = df[
                            df['author'].str.contains(author_search, case=False, na=False)
                        ]['author'].unique().tolist()
                        
                        if matching_authors:
                            favorite_authors = st.multiselect(
                                "Select from matching authors",
                                matching_authors,
                                default=[a for a in prefs.get('favorite_authors', []) if a in matching_authors],
                                key="pref_authors_search"
                            )
                        else:
                            st.info("No matching authors found")
                            favorite_authors = prefs.get('favorite_authors', [])
                    else:
                        favorite_authors = prefs.get('favorite_authors', [])
                
                prefs['favorite_authors'] = favorite_authors
            
            st.markdown("---")
            
            # Rating Preference
            if 'rating' in df.columns:
                st.markdown("#### Rating Preference")
                rating_range = st.slider(
                    "Preferred rating range",
                    0.0, 5.0,
                    prefs.get('rating_preference', (0.0, 5.0)),
                    0.5,
                    key="pref_rating"
                )
                prefs['rating_preference'] = rating_range
            
            # Price Range
            if 'price' in df.columns:
                st.markdown("#### Price Range")
                max_price = float(df['price'].max())
                price_range = st.slider(
                    "Preferred price range ($)",
                    0.0, max_price,
                    (0.0, min(prefs.get('price_range', (0.0, 100.0))[1], max_price)),
                    5.0,
                    key="pref_price"
                )
                prefs['price_range'] = price_range
            
            # Minimum Reviews
            review_cols = [col for col in df.columns if 'review' in col.lower()]
            if review_cols:
                st.markdown("#### Minimum Reviews")
                min_reviews = st.number_input(
                    "Minimum number of reviews",
                    min_value=0,
                    max_value=1000,
                    value=prefs.get('min_reviews', 0),
                    step=10,
                    key="pref_min_reviews"
                )
                prefs['min_reviews'] = min_reviews
        
        # TAB 2: Display Settings
        with tab2:
            st.markdown("### 🎨 Display Settings")
            
            # Items per page
            st.markdown("#### Pagination")
            items_per_page = st.selectbox(
                "Items per page",
                [5, 10, 20, 50, 100],
                index=[5, 10, 20, 50, 100].index(prefs.get('items_per_page', 10)),
                key="pref_items_per_page"
            )
            prefs['items_per_page'] = items_per_page
            
            # Default Sort
            st.markdown("#### Default Sorting")
            sort_options = [
                "Rating (High to Low)",
                "Rating (Low to High)",
                "Price (Low to High)",
                "Price (High to Low)",
                "Book Name (A-Z)",
                "Book Name (Z-A)"
            ]
            default_sort = st.selectbox(
                "Default sort order",
                sort_options,
                index=sort_options.index(prefs.get('default_sort', 'Rating (High to Low)')),
                key="pref_sort"
            )
            prefs['default_sort'] = default_sort
            
            # Show Descriptions
            st.markdown("#### Book Details")
            show_descriptions = st.checkbox(
                "Show book descriptions by default",
                value=prefs.get('show_descriptions', False),
                key="pref_show_desc"
            )
            prefs['show_descriptions'] = show_descriptions
            
            # Theme (placeholder for future implementation)
            st.markdown("#### Theme")
            theme = st.radio(
                "Color theme",
                ["Light", "Dark"],
                index=["Light", "Dark"].index(prefs.get('theme', 'Light')),
                key="pref_theme",
                horizontal=True
            )
            prefs['theme'] = theme
            st.info("Theme settings will be applied in a future update")
        
        # TAB 3: Recommendation Settings
        with tab3:
            st.markdown("### 🤖 Recommendation Settings")
            
            # Default Model
            st.markdown("#### Default Recommendation Model")
            model_options = ["Hybrid", "TF-IDF", "Content-Based", "Clustering"]
            recommendation_model = st.selectbox(
                "Preferred recommendation model",
                model_options,
                index=model_options.index(prefs.get('recommendation_model', 'Hybrid')),
                key="pref_model"
            )
            prefs['recommendation_model'] = recommendation_model
            
            # Model descriptions
            with st.expander("ℹ️ About Recommendation Models"):
                st.markdown("""
                **Hybrid (Recommended)**
                - Combines multiple algorithms for best results
                - Balances content similarity and popularity
                
                **TF-IDF**
                - Based on text analysis of book descriptions
                - Great for finding books with similar topics
                
                **Content-Based**
                - Uses book features (genre, author, etc.)
                - Good for finding books with similar characteristics
                
                **Clustering**
                - Groups similar books together
                - Discovers books in the same category
                """)
            
            st.markdown("---")
            
            # Model Weights (for Hybrid)
            if recommendation_model == "Hybrid":
                st.markdown("#### Hybrid Model Weights")
                st.caption("Adjust how much each algorithm contributes to recommendations")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tfidf_weight = st.slider(
                        "TF-IDF",
                        0.0, 1.0, 0.3, 0.1,
                        key="weight_tfidf"
                    )
                
                with col2:
                    content_weight = st.slider(
                        "Content-Based",
                        0.0, 1.0, 0.4, 0.1,
                        key="weight_content"
                    )
                
                with col3:
                    cluster_weight = st.slider(
                        "Clustering",
                        0.0, 1.0, 0.3, 0.1,
                        key="weight_cluster"
                    )
                
                total = tfidf_weight + content_weight + cluster_weight
                
                if abs(total - 1.0) > 0.01:
                    st.warning(f"⚠️ Weights sum to {total:.2f}. They should sum to 1.0")
                else:
                    st.success(f"✅ Weights are balanced (sum = {total:.2f})")
                
                prefs['model_weights'] = {
                    'tfidf': tfidf_weight,
                    'content': content_weight,
                    'clustering': cluster_weight
                }
        
        # TAB 4: Reading History
        with tab4:
            st.markdown("### 📚 Reading History")
            
            reading_history = prefs.get('reading_history', [])
            
            if reading_history:
                st.markdown(f"**You have {len(reading_history)} books in your reading history**")
                
                # Display reading history
                for i, book in enumerate(reading_history, 1):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"{i}. **{book['title']}** by {book['author']}")
                    
                    with col2:
                        if st.button("Remove", key=f"remove_book_{i}"):
                            reading_history.pop(i-1)
                            prefs['reading_history'] = reading_history
                            st.rerun()
                
                # Clear history button
                if st.button("🗑️ Clear Reading History", type="secondary"):
                    prefs['reading_history'] = []
                    st.success("Reading history cleared!")
                    st.rerun()
            else:
                st.info("No reading history yet. Books you view will appear here.")
            
            st.markdown("---")
            
            # Add to reading history
            st.markdown("#### Add Book to History")
            
            book_search = st.text_input("Search for a book to add", key="history_search")
            
            if book_search and 'book_name' in df.columns:
                matching_books = df[
                    df['book_name'].str.contains(book_search, case=False, na=False)
                ]['book_name'].head(10).tolist()
                
                if matching_books:
                    selected_book = st.selectbox(
                        "Select book",
                        matching_books,
                        key="history_book_select"
                    )
                    
                    if st.button("Add to History"):
                        book_data = df[df['book_name'] == selected_book].iloc[0]
                        
                        # Check if already in history
                        if not any(b['title'] == selected_book for b in reading_history):
                            reading_history.append({
                                'title': selected_book,
                                'author': book_data.get('author', 'Unknown'),
                                'genre': book_data.get('genre', 'Unknown')
                            })
                            prefs['reading_history'] = reading_history
                            st.success(f"Added '{selected_book}' to reading history!")
                            st.rerun()
                        else:
                            st.warning("This book is already in your history")
        
        st.markdown("---")
        
        # Save and Reset buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Save Preferences", type="primary", use_container_width=True):
                self.save_preferences(prefs)
        
        with col2:
            if st.button("🔄 Reset to Defaults", type="secondary", use_container_width=True):
                if st.session_state.get('confirm_reset', False):
                    default_prefs = self.load_preferences()
                    # Keep only defaults
                    for key in list(st.session_state[self.session_key].keys()):
                        if key in default_prefs:
                            st.session_state[self.session_key][key] = default_prefs[key]
                    st.success("Preferences reset to defaults!")
                    st.session_state['confirm_reset'] = False
                    st.rerun()
                else:
                    st.session_state['confirm_reset'] = True
                    st.warning("Click again to confirm reset")
    
    
    def apply_preferences_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on user preferences
        
        Args:
            df: Original dataframe
            
        Returns:
            Filtered dataframe
        """
        filtered_df = df.copy()
        prefs = st.session_state[self.session_key]
        
        # Apply rating filter
        if 'rating' in filtered_df.columns:
            min_rating, max_rating = prefs.get('rating_preference', (0.0, 5.0))
            filtered_df = filtered_df[
                (filtered_df['rating'] >= min_rating) &
                (filtered_df['rating'] <= max_rating)
            ]
        
        # Apply price filter
        if 'price' in filtered_df.columns:
            min_price, max_price = prefs.get('price_range', (0.0, 100.0))
            filtered_df = filtered_df[
                (filtered_df['price'] >= min_price) &
                (filtered_df['price'] <= max_price)
            ]
        
        # Apply genre exclusions
        if 'genre' in filtered_df.columns:
            exclude_genres = prefs.get('exclude_genres', [])
            if exclude_genres:
                filtered_df = filtered_df[~filtered_df['genre'].isin(exclude_genres)]
        
        # Apply minimum reviews filter
        review_cols = [col for col in filtered_df.columns if 'review' in col.lower()]
        if review_cols:
            min_reviews = prefs.get('min_reviews', 0)
            if min_reviews > 0:
                filtered_df = filtered_df[filtered_df[review_cols[0]] >= min_reviews]
        
        return filtered_df
    
    
    def get_personalized_recommendations_filter(self) -> Dict:
        """
        Get filter criteria based on user preferences
        
        Returns:
            Dictionary with filter criteria
        """
        prefs = st.session_state[self.session_key]
        
        return {
            'favorite_genres': prefs.get('favorite_genres', []),
            'favorite_authors': prefs.get('favorite_authors', []),
            'rating_range': prefs.get('rating_preference', (0.0, 5.0)),
            'price_range': prefs.get('price_range', (0.0, 100.0)),
            'exclude_genres': prefs.get('exclude_genres', []),
            'min_reviews': prefs.get('min_reviews', 0)
        }
    
    
    def add_to_reading_history(self, book_name: str, author: str, genre: str = "Unknown"):
        """
        Add a book to reading history
        
        Args:
            book_name: Name of the book
            author: Author name
            genre: Book genre
        """
        prefs = st.session_state[self.session_key]
        reading_history = prefs.get('reading_history', [])
        
        # Check if already exists
        if not any(b['title'] == book_name for b in reading_history):
            reading_history.append({
                'title': book_name,
                'author': author,
                'genre': genre
            })
            
            # Keep only last 50 books
            if len(reading_history) > 50:
                reading_history = reading_history[-50:]
            
            prefs['reading_history'] = reading_history
            st.session_state[self.session_key] = prefs
    
    
    def export_preferences(self) -> str:
        """
        Export preferences as JSON string
        
        Returns:
            JSON string of preferences
        """
        prefs = st.session_state[self.session_key]
        return json.dumps(prefs, indent=2)
    
    
    def import_preferences(self, json_string: str) -> bool:
        """
        Import preferences from JSON string
        
        Args:
            json_string: JSON string with preferences
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prefs = json.loads(json_string)
            st.session_state[self.session_key] = prefs
            self.save_preferences(prefs)
            return True
        except Exception as e:
            st.error(f"Error importing preferences: {e}")
            return False


def main():
    """Main function for testing the component"""
    import pandas as pd
    
    st.title("User Preferences Component - Demo")
    
    # Create sample dataframe
    df = pd.DataFrame({
        'book_name': ['Book ' + str(i) for i in range(100)],
        'author': ['Author ' + str(i % 20) for i in range(100)],
        'genre': ['Genre ' + str(i % 10) for i in range(100)],
        'rating': [3.5 + (i % 5) * 0.3 for i in range(100)],
        'price': [10 + (i % 10) * 5 for i in range(100)]
    })
    
    # Initialize preferences
    user_prefs = UserPreferences()
    
    # Render preferences panel
    user_prefs.render_preferences_panel(df)
    
    # Show current preferences
    with st.expander("📋 Current Preferences (JSON)"):
        st.json(st.session_state[user_prefs.session_key])


if __name__ == "__main__":
    main()
