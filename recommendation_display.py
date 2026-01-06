"""
Recommendation Display Component
Displays book recommendations in various formats
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional


class RecommendationDisplay:
    """Component for displaying book recommendations"""
    
    def __init__(self):
        """Initialize RecommendationDisplay"""
        pass
    
    
    def display_book_card(self, book_data: pd.Series, 
                         show_score: bool = False,
                         score: float = None,
                         show_description: bool = False,
                         card_style: str = "default"):
        """
        Display a single book in card format
        
        Args:
            book_data: Series with book information
            show_score: Whether to show recommendation score
            score: Recommendation score (0-1)
            show_description: Whether to show book description
            card_style: Style variant ('default', 'compact', 'detailed')
        """
        # Extract book information
        book_name = book_data.get('book_name', 'Unknown')
        author = book_data.get('author', 'Unknown')
        rating = book_data.get('rating', 0)
        genre = book_data.get('genre', 'Unknown')
        price = book_data.get('price', 0)
        description = book_data.get('description', '')
        
        # Container for card
        with st.container():
            if card_style == "compact":
                self._render_compact_card(
                    book_name, author, rating, genre, price, show_score, score
                )
            elif card_style == "detailed":
                self._render_detailed_card(
                    book_name, author, rating, genre, price, 
                    description, show_score, score
                )
            else:  # default
                self._render_default_card(
                    book_name, author, rating, genre, price, 
                    show_description, description, show_score, score
                )
    
    
    def _render_default_card(self, book_name, author, rating, genre, price,
                           show_description, description, show_score, score):
        """Render default card style"""
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 📖 {book_name}")
            st.markdown(f"**Author:** {author}")
            st.markdown(f"**Genre:** {genre}")
            
            # Rating stars
            stars = "⭐" * int(rating)
            empty_stars = "☆" * (5 - int(rating))
            st.markdown(f"**Rating:** {stars}{empty_stars} ({rating:.1f}/5.0)")
            
            if price > 0:
                st.markdown(f"**Price:** ${price:.2f}")
            
            if show_description and description:
                with st.expander("📄 Description"):
                    st.write(description)
        
        with col2:
            if show_score and score is not None:
                # Score gauge
                score_pct = score * 100
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <div style="font-size: 14px; opacity: 0.9;">Match Score</div>
                    <div style="font-size: 32px; font-weight: bold;">{score_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    
    def _render_compact_card(self, book_name, author, rating, genre, price,
                           show_score, score):
        """Render compact card style"""
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**{book_name}**")
            st.caption(f"by {author}")
        
        with col2:
            stars = "⭐" * int(rating)
            st.write(f"{stars} {rating:.1f}")
        
        with col3:
            if show_score and score is not None:
                st.markdown(f"🎯 **{score*100:.0f}%**")
            elif price > 0:
                st.write(f"${price:.2f}")
    
    
    def _render_detailed_card(self, book_name, author, rating, genre, price,
                            description, show_score, score):
        """Render detailed card style"""
        
        st.markdown(f"""
        <div style="
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        ">
            <h3 style="color: #2c3e50; margin-bottom: 10px;">📖 {book_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**✍️ Author:** {author}")
            st.markdown(f"**🎭 Genre:** {genre}")
            
            stars = "⭐" * int(rating)
            st.markdown(f"**Rating:** {stars} ({rating:.1f}/5.0)")
            
            if description:
                st.markdown(f"**📝 Description:**")
                st.write(description[:200] + "..." if len(description) > 200 else description)
        
        with col2:
            if price > 0:
                st.metric("💰 Price", f"${price:.2f}")
            
            if show_score and score is not None:
                st.metric("🎯 Match", f"{score*100:.0f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    
    def display_recommendation_grid(self, recommendations: pd.DataFrame,
                                   num_columns: int = 3,
                                   show_scores: bool = True):
        """
        Display recommendations in a grid layout
        
        Args:
            recommendations: DataFrame with recommendations
            num_columns: Number of columns in grid
            show_scores: Whether to show scores
        """
        if recommendations.empty:
            st.warning("No recommendations to display")
            return
        
        # Create grid
        for i in range(0, len(recommendations), num_columns):
            cols = st.columns(num_columns)
            
            for j, col in enumerate(cols):
                if i + j < len(recommendations):
                    with col:
                        row = recommendations.iloc[i + j]
                        score = row.get('similarity_score', row.get('hybrid_score', None))
                        
                        with st.container():
                            st.markdown(f"**{row['book_name'][:30]}...**" 
                                      if len(row['book_name']) > 30 
                                      else f"**{row['book_name']}**")
                            st.caption(f"by {row.get('author', 'Unknown')}")
                            
                            if 'rating' in row:
                                stars = "⭐" * int(row['rating'])
                                st.write(f"{stars} {row['rating']:.1f}")
                            
                            if show_scores and score is not None:
                                st.progress(score)
                                st.caption(f"Match: {score*100:.0f}%")
    
    
    def display_comparison_table(self, original_book: pd.Series,
                                recommendations: pd.DataFrame,
                                num_recommendations: int = 5):
        """
        Display comparison table between original book and recommendations
        
        Args:
            original_book: Series with original book data
            recommendations: DataFrame with recommendations
            num_recommendations: Number of recommendations to show
        """
        st.markdown("### 📊 Comparison Table")
        
        # Prepare comparison data
        comparison_data = {
            'Book': [original_book['book_name']] + 
                   recommendations['book_name'].head(num_recommendations).tolist(),
            'Author': [original_book.get('author', 'N/A')] + 
                     recommendations['author'].head(num_recommendations).tolist(),
            'Genre': [original_book.get('genre', 'N/A')] + 
                    recommendations['genre'].head(num_recommendations).tolist(),
            'Rating': [original_book.get('rating', 0)] + 
                     recommendations['rating'].head(num_recommendations).tolist()
        }
        
        if 'price' in original_book and 'price' in recommendations.columns:
            comparison_data['Price'] = [original_book['price']] + \
                                      recommendations['price'].head(num_recommendations).tolist()
        
        # Add match score for recommendations
        scores = ['Original'] + \
                [f"{score*100:.0f}%" for score in 
                 recommendations.head(num_recommendations).get('similarity_score', 
                 recommendations.head(num_recommendations).get('hybrid_score', [0]*num_recommendations))]
        comparison_data['Match'] = scores
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        st.dataframe(
            comparison_df.style.apply(
                lambda x: ['background-color: #e8f5e9' if i == 0 else '' 
                          for i in range(len(x))],
                axis=0
            ),
            use_container_width=True,
            hide_index=True
        )
    
    
    def display_recommendation_metrics(self, recommendations: pd.DataFrame):
        """
        Display metrics about recommendations
        
        Args:
            recommendations: DataFrame with recommendations
        """
        st.markdown("### 📈 Recommendation Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_rating = recommendations['rating'].mean() if 'rating' in recommendations else 0
            st.metric("Avg Rating", f"{avg_rating:.2f}⭐")
        
        with col2:
            if 'price' in recommendations.columns:
                avg_price = recommendations['price'].mean()
                st.metric("Avg Price", f"${avg_price:.2f}")
        
        with col3:
            unique_genres = recommendations['genre'].nunique() if 'genre' in recommendations else 0
            st.metric("Genres", unique_genres)
        
        with col4:
            unique_authors = recommendations['author'].nunique() if 'author' in recommendations else 0
            st.metric("Authors", unique_authors)
    
    
    def display_no_recommendations_message(self, book_name: str):
        """
        Display message when no recommendations found
        
        Args:
            book_name: Name of the book
        """
        st.warning(f"""
        ### 😕 No Recommendations Found
        
        We couldn't find recommendations for **"{book_name}"**.
        
        **Suggestions:**
        - Try a different book
        - Use the search feature to find similar books
        - Explore books by genre or author
        """)
