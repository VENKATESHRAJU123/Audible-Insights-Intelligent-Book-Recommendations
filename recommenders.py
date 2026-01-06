"""
Recommendation Models Module
Implements various recommendation algorithms
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFRecommender:
    """TF-IDF based content recommender"""
    
    def __init__(self):
        """Initialize TF-IDF Recommender"""
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.book_indices = {}
        self.books_df = None
        
        logger.info("TFIDFRecommender initialized")
    
    
    def fit(self, df: pd.DataFrame, text_column: str = 'combined_text'):
        """
        Fit the TF-IDF vectorizer
        
        Args:
            df: Dataframe with book data
            text_column: Column containing text to vectorize
        """
        logger.info("Fitting TF-IDF vectorizer...")
        
        self.books_df = df.copy()
        
        # Create combined text if not exists
        if text_column not in df.columns:
            text_cols = ['book_name', 'author', 'description', 'genre']
            df['combined_text'] = ''
            for col in text_cols:
                if col in df.columns:
                    df['combined_text'] += ' ' + df[col].astype(str)
            text_column = 'combined_text'
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(
            df[text_column].fillna('')
        )
        
        # Create book index mapping
        self.book_indices = {
            book: idx for idx, book in enumerate(df['book_name'])
        }
        
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    
    def get_recommendations(self, book_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get book recommendations based on content similarity
        
        Args:
            book_name: Name of the book
            top_n: Number of recommendations
            
        Returns:
            Dataframe with recommendations
        """
        if book_name not in self.book_indices:
            logger.warning(f"Book '{book_name}' not found")
            return pd.DataFrame()
        
        # Get book index
        idx = self.book_indices[book_name]
        
        # Calculate cosine similarity
        sim_scores = cosine_similarity(
            self.tfidf_matrix[idx:idx+1],
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar books (excluding itself)
        similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
        
        # Create recommendations dataframe
        recommendations = self.books_df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = sim_scores[similar_indices]
        
        # Select only columns that exist
        available_cols = ['book_name', 'similarity_score']
        for col in ['author', 'rating', 'genre', 'price', 'description']:
            if col in recommendations.columns:
                available_cols.append(col)
        
        return recommendations[available_cols]
    
    
    def save_model(self, path: str):
        """Save the TF-IDF vectorizer"""
        joblib.dump(self.vectorizer, path)
        logger.info(f"TF-IDF vectorizer saved to: {path}")
    
    
    def load_model(self, path: str):
        """Load the TF-IDF vectorizer"""
        self.vectorizer = joblib.load(path)
        logger.info(f"TF-IDF vectorizer loaded from: {path}")


class ContentBasedRecommender:
    """Content-based filtering recommender"""
    
    def __init__(self):
        """Initialize Content-Based Recommender"""
        self.feature_matrix = None
        self.similarity_matrix = None
        self.book_indices = {}
        self.books_df = None
        
        logger.info("ContentBasedRecommender initialized")
    
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the content-based model
        
        Args:
            df: Dataframe with features
        """
        logger.info("Fitting content-based model...")
        
        self.books_df = df.copy()
        
        # Select feature columns
        feature_cols = (
            [col for col in df.columns if col.startswith('tfidf_')] +
            [col for col in df.columns if col.endswith('_encoded')] +
            [col for col in df.columns if col.endswith('_scaled')]
        )
        
        if not feature_cols:
            logger.error("No feature columns found!")
            raise ValueError("No features available for content-based filtering")
        
        # Create feature matrix
        self.feature_matrix = df[feature_cols].fillna(0).values
        
        # Calculate similarity matrix
        logger.info("Calculating similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        # Create book index mapping
        self.book_indices = {
            book: idx for idx, book in enumerate(df['book_name'])
        }
        
        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
    
    
    def get_recommendations(self, book_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get recommendations based on content features
        
        Args:
            book_name: Name of the book
            top_n: Number of recommendations
            
        Returns:
            Dataframe with recommendations
        """
        if book_name not in self.book_indices:
            logger.warning(f"Book '{book_name}' not found")
            return pd.DataFrame()
        
        # Get book index
        idx = self.book_indices[book_name]
        
        # Get similarity scores
        sim_scores = self.similarity_matrix[idx]
        
        # Get top similar books (excluding itself)
        similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
        
        # Create recommendations dataframe
        recommendations = self.books_df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = sim_scores[similar_indices]
        
        # Select only columns that exist
        available_cols = ['book_name', 'similarity_score']
        for col in ['author', 'rating', 'genre', 'price', 'description']:
            if col in recommendations.columns:
                available_cols.append(col)
        
        return recommendations[available_cols]
    
    
    def get_recommendations_by_features(self, preferences: Dict, top_n: int = 10) -> pd.DataFrame:
        """
        Get recommendations based on user preferences
        
        Args:
            preferences: Dictionary of user preferences
            top_n: Number of recommendations
            
        Returns:
            Dataframe with recommendations
        """
        logger.info("Getting recommendations based on preferences...")
        
        # Filter based on preferences
        filtered_df = self.books_df.copy()
        
        if 'genre' in preferences and preferences['genre'] and 'genre' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['genre'].str.contains(preferences['genre'], case=False, na=False)
            ]
        
        if 'min_rating' in preferences and 'rating' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['rating'] >= preferences['min_rating']]
        
        if 'author' in preferences and preferences['author'] and 'author' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['author'].str.contains(preferences['author'], case=False, na=False)
            ]
        
        # Sort by rating if available
        if 'rating' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('rating', ascending=False).head(top_n)
        else:
            filtered_df = filtered_df.head(top_n)
        
        # Select only columns that exist
        available_cols = ['book_name']
        for col in ['author', 'rating', 'genre', 'price']:
            if col in filtered_df.columns:
                available_cols.append(col)
        
        return filtered_df[available_cols]
    
    
    def save_model(self, path: str):
        """Save the content-based model"""
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'book_indices': self.book_indices,
            'feature_matrix': self.feature_matrix
        }
        joblib.dump(model_data, path)
        logger.info(f"Content-based model saved to: {path}")
    
    
    def load_model(self, path: str):
        """Load the content-based model"""
        model_data = joblib.load(path)
        self.similarity_matrix = model_data['similarity_matrix']
        self.book_indices = model_data['book_indices']
        self.feature_matrix = model_data['feature_matrix']
        logger.info(f"Content-based model loaded from: {path}")


class ClusteringRecommender:
    """Clustering-based recommender"""
    
    def __init__(self, n_clusters: int = 8):
        """
        Initialize Clustering Recommender
        
        Args:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.cluster_labels = None
        self.books_df = None
        
        logger.info(f"ClusteringRecommender initialized with {n_clusters} clusters")
    
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the clustering model
        
        Args:
            df: Dataframe with features
        """
        logger.info("Fitting clustering model...")
        
        self.books_df = df.copy()
        
        # Select feature columns
        feature_cols = (
            [col for col in df.columns if col.startswith('tfidf_')] +
            [col for col in df.columns if col.endswith('_encoded')] +
            [col for col in df.columns if col.endswith('_scaled')]
        )
        
        if not feature_cols:
            logger.error("No feature columns found!")
            raise ValueError("No features available for clustering")
        
        # Create feature matrix
        X = df[feature_cols].fillna(0).values
        
        # Fit K-Means
        self.cluster_labels = self.kmeans.fit_predict(X)
        
        # Add cluster labels to dataframe
        self.books_df['cluster'] = self.cluster_labels
        
        logger.info(f"Clustering completed with {self.n_clusters} clusters")
        
        # Print cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logger.info(f"Cluster {cluster_id}: {count} books")
    
    
    def get_recommendations(self, book_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get recommendations from the same cluster
        
        Args:
            book_name: Name of the book
            top_n: Number of recommendations
            
        Returns:
            Dataframe with recommendations
        """
        # Find the book
        book_match = self.books_df[self.books_df['book_name'] == book_name]
        
        if book_match.empty:
            logger.warning(f"Book '{book_name}' not found")
            return pd.DataFrame()
        
        # Get cluster of the book
        book_cluster = book_match.iloc[0]['cluster']
        
        # Get books from same cluster
        cluster_books = self.books_df[
            (self.books_df['cluster'] == book_cluster) &
            (self.books_df['book_name'] != book_name)
        ]
        
        # Sort by rating if available, otherwise just return top N
        if 'rating' in cluster_books.columns:
            recommendations = cluster_books.sort_values('rating', ascending=False).head(top_n)
        else:
            recommendations = cluster_books.head(top_n)
        
        # Select only columns that exist
        available_cols = ['book_name', 'cluster']
        for col in ['author', 'rating', 'genre', 'price']:
            if col in recommendations.columns:
                available_cols.append(col)
        
        return recommendations[available_cols]
    
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each cluster
        
        Returns:
            Dataframe with cluster summaries
        """
        summaries = []
        
        for cluster_id in range(self.n_clusters):
            cluster_books = self.books_df[self.books_df['cluster'] == cluster_id]
            
            summary = {
                'cluster_id': cluster_id,
                'num_books': len(cluster_books)
            }
            
            # Add rating if available
            if 'rating' in cluster_books.columns:
                summary['avg_rating'] = cluster_books['rating'].mean()
            
            # Add top genre if available
            if 'genre' in cluster_books.columns:
                top_genre_series = cluster_books['genre'].mode()
                summary['top_genre'] = top_genre_series[0] if not top_genre_series.empty else 'Unknown'
            
            # Add top author if available
            if 'author' in cluster_books.columns:
                top_author_series = cluster_books['author'].mode()
                summary['top_author'] = top_author_series[0] if not top_author_series.empty else 'Unknown'
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    
    def save_model(self, path: str):
        """Save the clustering model"""
        joblib.dump(self.kmeans, path)
        logger.info(f"Clustering model saved to: {path}")
    
    
    def load_model(self, path: str):
        """Load the clustering model"""
        self.kmeans = joblib.load(path)
        logger.info(f"Clustering model loaded from: {path}")


class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches"""
    
    def __init__(self):
        """Initialize Hybrid Recommender"""
        self.tfidf_recommender = TFIDFRecommender()
        self.content_recommender = ContentBasedRecommender()
        self.clustering_recommender = ClusteringRecommender()
        self.books_df = None
        
        logger.info("HybridRecommender initialized")
    
    
    def fit(self, df: pd.DataFrame):
        """
        Fit all sub-models
        
        Args:
            df: Dataframe with book data and features
        """
        logger.info("Fitting hybrid recommender...")
        
        self.books_df = df.copy()
        
        # Fit all models
        self.tfidf_recommender.fit(df)
        self.content_recommender.fit(df)
        self.clustering_recommender.fit(df)
        
        logger.info("Hybrid recommender fitted successfully")
    
    
    def get_recommendations(self, book_name: str, top_n: int = 10, 
                          weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Get hybrid recommendations
        
        Args:
            book_name: Name of the book
            top_n: Number of recommendations
            weights: Dictionary of weights for each model
            
        Returns:
            Dataframe with recommendations
        """
        if weights is None:
            weights = {
                'tfidf': 0.3,
                'content': 0.4,
                'clustering': 0.3
            }
        
        logger.info(f"Getting hybrid recommendations for '{book_name}'")
        
        # Get recommendations from each model
        tfidf_recs = self.tfidf_recommender.get_recommendations(book_name, top_n * 2)
        content_recs = self.content_recommender.get_recommendations(book_name, top_n * 2)
        cluster_recs = self.clustering_recommender.get_recommendations(book_name, top_n * 2)
        
        # Combine recommendations
        all_recs = {}
        
        # Add TF-IDF recommendations
        for idx, row in tfidf_recs.iterrows():
            book = row['book_name']
            if book not in all_recs:
                all_recs[book] = {'score': 0, 'data': row}
            all_recs[book]['score'] += row.get('similarity_score', 0) * weights['tfidf']
        
        # Add content-based recommendations
        for idx, row in content_recs.iterrows():
            book = row['book_name']
            if book not in all_recs:
                all_recs[book] = {'score': 0, 'data': row}
            all_recs[book]['score'] += row.get('similarity_score', 0) * weights['content']
        
        # Add clustering recommendations
        for idx, row in cluster_recs.iterrows():
            book = row['book_name']
            if book not in all_recs:
                all_recs[book] = {'score': 0, 'data': row}
            # Normalize rating to 0-1 scale for clustering if available
            if 'rating' in row and pd.notna(row['rating']):
                normalized_rating = row['rating'] / 5.0
            else:
                normalized_rating = 0.5  # Default neutral score
            all_recs[book]['score'] += normalized_rating * weights['clustering']
        
        # Sort by combined score
        sorted_books = sorted(all_recs.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Create recommendations dataframe
        recommendations = []
        for book, info in sorted_books[:top_n]:
            rec = info['data'].to_dict()
            rec['hybrid_score'] = info['score']
            recommendations.append(rec)
        
        if not recommendations:
            return pd.DataFrame()
        
        recommendations_df = pd.DataFrame(recommendations)
        
        # Select only columns that exist
        available_cols = ['book_name', 'hybrid_score']
        for col in ['author', 'rating', 'genre', 'price', 'description']:
            if col in recommendations_df.columns:
                available_cols.append(col)
        
        return recommendations_df[available_cols]
    
    
    def save_model(self, path: str):
        """Save the hybrid model"""
        model_data = {
            'tfidf_recommender': self.tfidf_recommender,
            'content_recommender': self.content_recommender,
            'clustering_recommender': self.clustering_recommender
        }
        joblib.dump(model_data, path)
        logger.info(f"Hybrid recommender saved to: {path}")
    
    
    def load_model(self, path: str):
        """Load the hybrid model"""
        model_data = joblib.load(path)
        self.tfidf_recommender = model_data['tfidf_recommender']
        self.content_recommender = model_data['content_recommender']
        self.clustering_recommender = model_data['clustering_recommender']
        logger.info(f"Hybrid recommender loaded from: {path}")


def main():
    """Main execution function for testing"""
    
    # Load data
    logger.info("Loading clustered data...")
    df = pd.read_csv('data/processed/clustered_data.csv')
    
    print("\n" + "="*60)
    print("RECOMMENDATION MODELS - TRAINING & TESTING")
    print("="*60)
    
    # Test each model
    print("\n📚 1. TF-IDF Recommender")
    print("-" * 60)
    tfidf_rec = TFIDFRecommender()
    tfidf_rec.fit(df)
    
    print("\n📊 2. Content-Based Recommender")
    print("-" * 60)
    content_rec = ContentBasedRecommender()
    content_rec.fit(df)
    
    print("\n🎯 3. Clustering Recommender")
    print("-" * 60)
    cluster_rec = ClusteringRecommender(n_clusters=8)
    cluster_rec.fit(df)
    
    print("\n🔥 4. Hybrid Recommender")
    print("-" * 60)
    hybrid_rec = HybridRecommender()
    hybrid_rec.fit(df)
    
    # Get sample recommendations
    if len(df) > 0:
        sample_book = df.iloc[0]['book_name']
        print(f"\nSample Recommendations for: '{sample_book}'")
        print("-" * 60)
        
        recs = hybrid_rec.get_recommendations(sample_book, top_n=5)
        print(recs)
    
    logger.info("\n✅ All models trained successfully!")


if __name__ == "__main__":
    main()
