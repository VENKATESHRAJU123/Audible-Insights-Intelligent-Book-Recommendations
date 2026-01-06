"""
Search Interface Component
Provides advanced search and filter functionality
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple


class SearchInterface:
    """Component for searching and filtering books"""
    
    def __init__(self):
        """Initialize SearchInterface"""
        self.filters = {}
    
    
    def render_search_bar(self, df: pd.DataFrame, 
                         placeholder: str = "Search by book name...") -> str:
        """
        Render search input bar
        
        Args:
            df: DataFrame to search
            placeholder: Placeholder text
            
        Returns:
            Search query string
        """
        search_query = st.text_input(
            "🔍 Search Books",
            placeholder=placeholder,
            help="Enter book name or keywords"
        )
        
        return search_query
    
    
    def render_filter_sidebar(self, df: pd.DataFrame) -> Dict:
        """
        Render comprehensive filter sidebar
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Dictionary of filter values
        """
        st.sidebar.markdown("## 🔎 Advanced Filters")
        
        filters = {}
        
        # Genre filter
        if 'genre' in df.columns:
            st.sidebar.markdown("### 📚 Genre")
            genres = ['All'] + sorted(df['genre'].dropna().unique().tolist())
            filters['genre'] = st.sidebar.selectbox(
                "Select Genre",
                genres,
                key="genre_filter"
            )
        
        # Author filter
        if 'author' in df.columns:
            st.sidebar.markdown("### ✍️ Author")
            
            # Option for top authors or all
            author_option = st.sidebar.radio(
                "Author Selection",
                ["Top Authors", "All Authors", "Search Author"],
                key="author_option"
            )
            
            if author_option == "Top Authors":
                top_authors = ['All'] + df['author'].value_counts().head(20).index.tolist()
                filters['author'] = st.sidebar.selectbox(
                    "Select Author",
                    top_authors,
                    key="author_filter_top"
                )
            elif author_option == "All Authors":
                all_authors = ['All'] + sorted(df['author'].dropna().unique().tolist())
                filters['author'] = st.sidebar.selectbox(
                    "Select Author",
                    all_authors,
                    key="author_filter_all"
                )
            else:  # Search Author
                author_search = st.sidebar.text_input(
                    "Search Author",
                    key="author_search"
                )
                if author_search:
                    matching_authors = df[
                        df['author'].str.contains(author_search, case=False, na=False)
                    ]['author'].unique().tolist()
                    
                    if matching_authors:
                        filters['author'] = st.sidebar.selectbox(
                            "Matching Authors",
                            ['All'] + matching_authors,
                            key="author_filter_search"
                        )
                    else:
                        st.sidebar.warning("No matching authors found")
                        filters['author'] = 'All'
                else:
                    filters['author'] = 'All'
        
        # Rating filter
        if 'rating' in df.columns:
            st.sidebar.markdown("### ⭐ Rating")
            
            min_rating, max_rating = st.sidebar.slider(
                "Rating Range",
                min_value=0.0,
                max_value=5.0,
                value=(0.0, 5.0),
                step=0.5,
                key="rating_filter"
            )
            filters['rating_range'] = (min_rating, max_rating)
        
        # Price filter
        if 'price' in df.columns:
            st.sidebar.markdown("### 💰 Price")
            
            price_max = float(df['price'].max())
            
            max_price = st.sidebar.slider(
                "Maximum Price ($)",
                min_value=0.0,
                max_value=price_max,
                value=price_max,
                step=5.0,
                key="price_filter"
            )
            filters['max_price'] = max_price
        
        # Review count filter (if available)
        review_cols = [col for col in df.columns if 'review' in col.lower()]
        if review_cols:
            st.sidebar.markdown("### 📝 Reviews")
            
            min_reviews = st.sidebar.number_input(
                "Minimum Reviews",
                min_value=0,
                max_value=int(df[review_cols[0]].max()),
                value=0,
                step=10,
                key="review_filter"
            )
            filters['min_reviews'] = min_reviews
            filters['review_column'] = review_cols[0]
        
        # Cluster filter (if available)
        if 'cluster_kmeans' in df.columns:
            st.sidebar.markdown("### 🎯 Book Cluster")
            
            clusters = sorted(df['cluster_kmeans'].dropna().unique())
            selected_clusters = st.sidebar.multiselect(
                "Select Clusters",
                clusters,
                key="cluster_filter",
                help="Books grouped by similarity"
            )
            filters['clusters'] = selected_clusters
        
        self.filters = filters
        return filters
    
    
    def apply_filters(self, df: pd.DataFrame, 
                     filters: Dict, 
                     search_query: str = "") -> pd.DataFrame:
        """
        Apply all filters to dataframe
        
        Args:
            df: Original dataframe
            filters: Dictionary of filter values
            search_query: Search query string
            
        Returns:
            Filtered dataframe
        """
        filtered_df = df.copy()
        
        # Apply search query
        if search_query:
            filtered_df = filtered_df[
                filtered_df['book_name'].str.contains(search_query, case=False, na=False)
            ]
        
        # Apply genre filter
        if filters.get('genre') and filters['genre'] != 'All':
            filtered_df = filtered_df[filtered_df['genre'] == filters['genre']]
        
        # Apply author filter
        if filters.get('author') and filters['author'] != 'All':
            filtered_df = filtered_df[filtered_df['author'] == filters['author']]
        
        # Apply rating filter
        if 'rating_range' in filters and 'rating' in filtered_df.columns:
            min_rating, max_rating = filters['rating_range']
            filtered_df = filtered_df[
                (filtered_df['rating'] >= min_rating) &
                (filtered_df['rating'] <= max_rating)
            ]
        
        # Apply price filter
        if 'max_price' in filters and 'price' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['price'] <= filters['max_price']]
        
        # Apply review filter
        if 'min_reviews' in filters and 'review_column' in filters:
            review_col = filters['review_column']
            if review_col in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df[review_col] >= filters['min_reviews']
                ]
        
        # Apply cluster filter
        if filters.get('clusters') and 'cluster_kmeans' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['cluster_kmeans'].isin(filters['clusters'])
            ]
        
        return filtered_df
    
    
    def render_sort_options(self, df: pd.DataFrame) -> Tuple[str, bool]:
        """
        Render sorting options
        
        Args:
            df: DataFrame to sort
            
        Returns:
            Tuple of (sort_column, ascending)
        """
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Determine available sort options
            sort_options = {
                "Book Name (A-Z)": ("book_name", True),
                "Book Name (Z-A)": ("book_name", False)
            }
            
            if 'rating' in df.columns:
                sort_options.update({
                    "Rating (High to Low)": ("rating", False),
                    "Rating (Low to High)": ("rating", True)
                })
            
            if 'price' in df.columns:
                sort_options.update({
                    "Price (Low to High)": ("price", True),
                    "Price (High to Low)": ("price", False)
                })
            
            review_cols = [col for col in df.columns if 'review' in col.lower()]
            if review_cols:
                sort_options.update({
                    "Most Reviews": (review_cols[0], False),
                    "Least Reviews": (review_cols[0], True)
                })
            
            selected_sort = st.selectbox(
                "Sort by",
                list(sort_options.keys()),
                key="sort_option"
            )
            
            sort_column, ascending = sort_options[selected_sort]
        
        with col2:
            items_per_page = st.selectbox(
                "Items per page",
                [10, 20, 50, 100],
                index=0,
                key="items_per_page"
            )
        
        return sort_column, ascending, items_per_page
    
    
    def render_pagination(self, total_items: int, 
                         items_per_page: int) -> Tuple[int, int]:
        """
        Render pagination controls
        
        Args:
            total_items: Total number of items
            items_per_page: Items per page
            
        Returns:
            Tuple of (start_idx, end_idx)
        """
        total_pages = max(1, (total_items - 1) // items_per_page + 1)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="page_number"
            )
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_items} results")
        
        return start_idx, end_idx
    
    
    def render_filter_summary(self, filters: Dict, results_count: int):
        """
        Display active filter summary
        
        Args:
            filters: Dictionary of active filters
            results_count: Number of results
        """
        active_filters = []
        
        if filters.get('genre') and filters['genre'] != 'All':
            active_filters.append(f"Genre: {filters['genre']}")
        
        if filters.get('author') and filters['author'] != 'All':
            active_filters.append(f"Author: {filters['author']}")
        
        if 'rating_range' in filters:
            min_r, max_r = filters['rating_range']
            if min_r > 0 or max_r < 5:
                active_filters.append(f"Rating: {min_r}-{max_r}")
        
        if 'max_price' in filters:
            active_filters.append(f"Price ≤ ${filters['max_price']:.0f}")
        
        if filters.get('min_reviews', 0) > 0:
            active_filters.append(f"Reviews ≥ {filters['min_reviews']}")
        
        if filters.get('clusters'):
            active_filters.append(f"Clusters: {', '.join(map(str, filters['clusters']))}")
        
        if active_filters:
            st.info(f"**Active Filters:** {' | '.join(active_filters)} → **{results_count} results**")
        else:
            st.info(f"**{results_count} results** (no filters applied)")
    
    
    def render_export_options(self, df: pd.DataFrame, filename_prefix: str = "books"):
        """
        Render data export options
        
        Args:
            df: DataFrame to export
            filename_prefix: Prefix for filename
        """
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"{filename_prefix}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel export (if openpyxl available)
            try:
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Books')
                
                st.download_button(
                    label="📊 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"{filename_prefix}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.button(
                    "📊 Download Excel",
                    disabled=True,
                    help="Install openpyxl to enable Excel export",
                    use_container_width=True
                )
        
        with col3:
            # JSON export
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="🔗 Download JSON",
                data=json_str,
                file_name=f"{filename_prefix}.json",
                mime="application/json",
                use_container_width=True
            )
