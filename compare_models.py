"""
Model Comparison Script
Compare performance of different recommendation models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.recommenders import (
    TFIDFRecommender,
    ContentBasedRecommender,
    ClusteringRecommender,
    HybridRecommender
)


def compare_models():
    """Compare all recommendation models"""
    
    # Load data
    df = pd.read_csv('data/processed/clustered_data.csv')
    
    # Initialize models
    models = {
        'TF-IDF': TFIDFRecommender(),
        'Content-Based': ContentBasedRecommender(),
        'Clustering': ClusteringRecommender(n_clusters=8),
        'Hybrid': HybridRecommender()
    }
    
    # Train all models
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(df)
    
    # Test with multiple books
    test_books = df.sample(min(5, len(df)))['book_name'].tolist()
    
    results = []
    
    for book in test_books:
        print(f"\nTesting with: '{book}'")
        
        for model_name, model in models.items():
            recs = model.get_recommendations(book, top_n=10)
            
            if not recs.empty:
                avg_rating = recs['rating'].mean() if 'rating' in recs.columns else 0
                
                results.append({
                    'Book': book,
                    'Model': model_name,
                    'Num_Recommendations': len(recs),
                    'Avg_Rating': avg_rating
                })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    
    # Aggregate results
    summary = comparison_df.groupby('Model').agg({
        'Num_Recommendations': 'mean',
        'Avg_Rating': 'mean'
    }).round(2)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(summary)
    
    # Save comparison
    comparison_df.to_csv('outputs/reports/model_comparison.csv', index=False)
    print("\nComparison saved to: outputs/reports/model_comparison.csv")


if __name__ == "__main__":
    compare_models()
