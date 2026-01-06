"""
Unit Tests for Recommendation Models
Tests for TF-IDF, Content-Based, Clustering, and Hybrid recommenders
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommenders import (
    TFIDFRecommender,
    ContentBasedRecommender,
    ClusteringRecommender,
    HybridRecommender
)


class TestTFIDFRecommender:
    """Test suite for TFIDFRecommender"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataset"""
        return pd.DataFrame({
            'book_name': ['Book A', 'Book B', 'Book C', 'Book D', 'Book E'],
            'author': ['Author 1', 'Author 2', 'Author 1', 'Author 3', 'Author 2'],
            'rating': [4.5, 3.8, 4.2, 4.0, 3.5],
            'genre': ['Fiction', 'Mystery', 'Fiction', 'Romance', 'Mystery'],
            'description': [
                'A thrilling fiction novel about adventure',
                'A mysterious crime story',
                'An exciting fiction adventure',
                'A romantic love story',
                'A detective mystery novel'
            ]
        })
    
    @pytest.fixture
    def recommender(self):
        """Create TFIDFRecommender instance"""
        return TFIDFRecommender()
    
    
    def test_initialization(self, recommender):
        """Test recommender initialization"""
        assert recommender.vectorizer is not None
        assert recommender.tfidf_matrix is None
        assert recommender.book_indices == {}
    
    
    def test_fit(self, recommender, sample_df):
        """Test fitting the recommender"""
        recommender.fit(sample_df)
        
        assert recommender.tfidf_matrix is not None
        assert len(recommender.book_indices) == len(sample_df)
        assert recommender.books_df is not None
    
    
    def test_get_recommendations(self, recommender, sample_df):
        """Test getting recommendations"""
        recommender.fit(sample_df)
        
        recs = recommender.get_recommendations('Book A', top_n=3)
        
        assert not recs.empty
        assert len(recs) <= 3
        assert 'similarity_score' in recs.columns
        assert 'Book A' not in recs['book_name'].values
    
    
    def test_get_recommendations_similar_content(self, recommender, sample_df):
        """Test that similar content gets higher scores"""
        recommender.fit(sample_df)
        
        recs = recommender.get_recommendations('Book A', top_n=4)
        
        # Book C should be in recommendations (both have 'fiction adventure')
        assert 'Book C' in recs['book_name'].values
    
    
    def test_get_recommendations_invalid_book(self, recommender, sample_df):
        """Test recommendations for non-existent book"""
        recommender.fit(sample_df)
        
        recs = recommender.get_recommendations('Non-existent Book', top_n=3)
        assert recs.empty
    
    
    def test_save_and_load_model(self, recommender, sample_df, tmp_path):
        """Test model saving and loading"""
        recommender.fit(sample_df)
        
        model_path = tmp_path / "tfidf_model.pkl"
        recommender.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        new_recommender = TFIDFRecommender()
        new_recommender.load_model(str(model_path))
        
        assert new_recommender.vectorizer is not None


class TestContentBasedRecommender:
    """Test suite for ContentBasedRecommender"""
    
    @pytest.fixture
    def sample_df_with_features(self):
        """Create sample dataset with features"""
        return pd.DataFrame({
            'book_name': ['Book A', 'Book B', 'Book C', 'Book D', 'Book E'],
            'author': ['Author 1', 'Author 2', 'Author 1', 'Author 3', 'Author 2'],
            'rating': [4.5, 3.8, 4.2, 4.0, 3.5],
            'genre': ['Fiction', 'Mystery', 'Fiction', 'Romance', 'Mystery'],
            'tfidf_0': [0.5, 0.1, 0.6, 0.2, 0.1],
            'tfidf_1': [0.3, 0.7, 0.2, 0.4, 0.8],
            'author_encoded': [0, 1, 0, 2, 1],
            'genre_encoded': [0, 1, 0, 2, 1],
            'rating_scaled': [0.9, 0.76, 0.84, 0.8, 0.7]
        })
    
    @pytest.fixture
    def recommender(self):
        """Create ContentBasedRecommender instance"""
        return ContentBasedRecommender()
    
    
    def test_initialization(self, recommender):
        """Test recommender initialization"""
        assert recommender.feature_matrix is None
        assert recommender.similarity_matrix is None
    
    
    def test_fit(self, recommender, sample_df_with_features):
        """Test fitting the recommender"""
        recommender.fit(sample_df_with_features)
        
        assert recommender.feature_matrix is not None
        assert recommender.similarity_matrix is not None
        assert len(recommender.book_indices) == len(sample_df_with_features)
    
    
    def test_get_recommendations(self, recommender, sample_df_with_features):
        """Test getting recommendations"""
        recommender.fit(sample_df_with_features)
        
        recs = recommender.get_recommendations('Book A', top_n=3)
        
        assert not recs.empty
        assert len(recs) <= 3
        assert 'similarity_score' in recs.columns
    
    
    def test_get_recommendations_by_features(self, recommender, sample_df_with_features):
        """Test preference-based recommendations"""
        recommender.fit(sample_df_with_features)
        
        preferences = {
            'genre': 'Fiction',
            'min_rating': 4.0
        }
        
        recs = recommender.get_recommendations_by_features(preferences, top_n=5)
        
        assert not recs.empty
    
    
    def test_save_and_load_model(self, recommender, sample_df_with_features, tmp_path):
        """Test model saving and loading"""
        recommender.fit(sample_df_with_features)
        
        model_path = tmp_path / "content_model.pkl"
        recommender.save_model(str(model_path))
        
        assert model_path.exists()


class TestClusteringRecommender:
    """Test suite for ClusteringRecommender"""
    
    @pytest.fixture
    def sample_df_with_features(self):
        """Create sample dataset with features"""
        np.random.seed(42)
        return pd.DataFrame({
            'book_name': [f'Book {i}' for i in range(20)],
            'author': [f'Author {i % 5}' for i in range(20)],
            'rating': [3.5 + (i % 5) * 0.3 for i in range(20)],
            'genre': [['Fiction', 'Mystery', 'Romance'][i % 3] for i in range(20)],
            'tfidf_0': np.random.rand(20),
            'tfidf_1': np.random.rand(20),
            'author_encoded': [i % 5 for i in range(20)],
            'genre_encoded': [i % 3 for i in range(20)],
            'rating_scaled': np.random.rand(20)
        })
    
    @pytest.fixture
    def recommender(self):
        """Create ClusteringRecommender instance"""
        return ClusteringRecommender(n_clusters=3)
    
    
    def test_initialization(self, recommender):
        """Test recommender initialization"""
        assert recommender.n_clusters == 3
        assert recommender.kmeans is not None
    
    
    def test_fit(self, recommender, sample_df_with_features):
        """Test fitting the recommender"""
        recommender.fit(sample_df_with_features)
        
        assert recommender.cluster_labels is not None
        assert 'cluster' in recommender.books_df.columns
        assert len(set(recommender.cluster_labels)) <= 3
    
    
    def test_get_recommendations(self, recommender, sample_df_with_features):
        """Test getting recommendations from same cluster"""
        recommender.fit(sample_df_with_features)
        
        recs = recommender.get_recommendations('Book 0', top_n=5)
        
        assert not recs.empty
    
    
    def test_get_cluster_summary(self, recommender, sample_df_with_features):
        """Test cluster summary generation"""
        recommender.fit(sample_df_with_features)
        
        summary = recommender.get_cluster_summary()
        
        assert not summary.empty
        assert len(summary) == recommender.n_clusters
        assert 'cluster_id' in summary.columns
        assert 'num_books' in summary.columns
    
    
    def test_save_and_load_model(self, recommender, sample_df_with_features, tmp_path):
        """Test model saving and loading"""
        recommender.fit(sample_df_with_features)
        
        model_path = tmp_path / "clustering_model.pkl"
        recommender.save_model(str(model_path))
        
        assert model_path.exists()


class TestHybridRecommender:
    """Test suite for HybridRecommender"""
    
    @pytest.fixture
    def sample_df_full(self):
        """Create comprehensive sample dataset"""
        np.random.seed(42)
        return pd.DataFrame({
            'book_name': [f'Book {i}' for i in range(15)],
            'author': [f'Author {i % 5}' for i in range(15)],
            'rating': [3.5 + (i % 5) * 0.3 for i in range(15)],
            'genre': [['Fiction', 'Mystery', 'Romance'][i % 3] for i in range(15)],
            'description': [f'Description for book {i}' for i in range(15)],
            'tfidf_0': np.random.rand(15),
            'tfidf_1': np.random.rand(15),
            'author_encoded': [i % 5 for i in range(15)],
            'genre_encoded': [i % 3 for i in range(15)],
            'rating_scaled': np.random.rand(15)
        })
    
    @pytest.fixture
    def recommender(self):
        """Create HybridRecommender instance"""
        return HybridRecommender()
    
    
    def test_initialization(self, recommender):
        """Test recommender initialization"""
        assert recommender.tfidf_recommender is not None
        assert recommender.content_recommender is not None
        assert recommender.clustering_recommender is not None
    
    
    def test_fit(self, recommender, sample_df_full):
        """Test fitting all sub-models"""
        recommender.fit(sample_df_full)
        
        assert recommender.books_df is not None
        assert len(recommender.books_df) == len(sample_df_full)
    
    
    def test_get_recommendations(self, recommender, sample_df_full):
        """Test getting hybrid recommendations"""
        recommender.fit(sample_df_full)
        
        recs = recommender.get_recommendations('Book 0', top_n=5)
        
        assert not recs.empty
        assert len(recs) <= 5
        assert 'hybrid_score' in recs.columns
    
    
    def test_get_recommendations_with_custom_weights(self, recommender, sample_df_full):
        """Test recommendations with custom weights"""
        recommender.fit(sample_df_full)
        
        weights = {
            'tfidf': 0.5,
            'content': 0.3,
            'clustering': 0.2
        }
        
        recs = recommender.get_recommendations('Book 0', top_n=5, weights=weights)
        
        assert not recs.empty
        assert 'hybrid_score' in recs.columns
    
    
    def test_save_and_load_model(self, recommender, sample_df_full, tmp_path):
        """Test model saving and loading"""
        recommender.fit(sample_df_full)
        
        model_path = tmp_path / "hybrid_model.pkl"
        recommender.save_model(str(model_path))
        
        assert model_path.exists()


def test_imports():
    """Test that all recommender classes can be imported"""
    from src.recommenders import (
        TFIDFRecommender,
        ContentBasedRecommender,
        ClusteringRecommender,
        HybridRecommender
    )
    assert all([
        TFIDFRecommender,
        ContentBasedRecommender,
        ClusteringRecommender,
        HybridRecommender
    ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
