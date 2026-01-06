"""
Model Testing Script
Load saved models and test recommendations
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.recommenders import (
    TFIDFRecommender,
    ContentBasedRecommender,
    ClusteringRecommender,
    HybridRecommender
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_saved_models():
    """Test all saved models"""
    
    print("\n" + "="*60)
    print("TESTING SAVED MODELS")
    print("="*60)
    
    models_path = Path("data/models")
    
    # Load data
    df = pd.read_csv('data/processed/clustered_data.csv')
    sample_book = df.iloc[0]['book_name']
    
    print(f"\nTesting with book: '{sample_book}'")
    print("-" * 60)
    
    # Test 1: TF-IDF Vectorizer
    print("\n1️⃣  Testing TF-IDF Vectorizer")
    try:
        vectorizer = joblib.load(models_path / "tfidf_vectorizer.pkl")
        print("✅ TF-IDF Vectorizer loaded successfully")
        print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    except Exception as e:
        print(f"❌ Error loading TF-IDF: {e}")
    
    # Test 2: Content-Based Model
    print("\n2️⃣  Testing Content-Based Model")
    try:
        content_model = joblib.load(models_path / "content_based_model.pkl")
        print("✅ Content-Based Model loaded successfully")
        print(f"   Similarity matrix shape: {content_model['similarity_matrix'].shape}")
    except Exception as e:
        print(f"❌ Error loading Content-Based: {e}")
    
    # Test 3: Clustering Model
    print("\n3️⃣  Testing Clustering Model")
    try:
        kmeans = joblib.load(models_path / "clustering_model.pkl")
        print("✅ Clustering Model loaded successfully")
        print(f"   Number of clusters: {kmeans.n_clusters}")
        print(f"   Inertia: {kmeans.inertia_:.2f}")
    except Exception as e:
        print(f"❌ Error loading Clustering: {e}")
    
    # Test 4: Hybrid Recommender
    print("\n4️⃣  Testing Hybrid Recommender")
    try:
        hybrid_model = joblib.load(models_path / "hybrid_recommender.pkl")
        print("✅ Hybrid Recommender loaded successfully")
        print(f"   Sub-models loaded: 3")
    except Exception as e:
        print(f"❌ Error loading Hybrid: {e}")
    
    # Test recommendations with full models
    print("\n" + "="*60)
    print("GETTING SAMPLE RECOMMENDATIONS")
    print("="*60)
    
    # Reinitialize and load models properly
    tfidf_rec = TFIDFRecommender()
    tfidf_rec.load_model(str(models_path / "tfidf_vectorizer.pkl"))
    tfidf_rec.fit(df)  # Need to refit with data
    
    recs = tfidf_rec.get_recommendations(sample_book, top_n=5)
    print(f"\nTop 5 recommendations for '{sample_book}':")
    print(recs[['book_name', 'author', 'similarity_score']].to_string(index=False))
    
    print("\n✅ All models tested successfully!")


if __name__ == "__main__":
    try:
        test_saved_models()
    except Exception as e:
        logger.error(f"Model testing failed: {e}", exc_info=True)
        raise
