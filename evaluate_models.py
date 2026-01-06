"""
Model Evaluation Script
Evaluates all trained recommendation models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import RecommenderEvaluator
from src.recommenders import (
    TFIDFRecommender,
    ContentBasedRecommender,
    ClusteringRecommender,
    HybridRecommender
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_set(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Create train/test split for evaluation
    
    Args:
        df: Full dataset
        test_size: Proportion of test set
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("Creating train/test split...")
    
    # Shuffle and split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * (1 - test_size))
    
    train_df = df_shuffled[:split_idx]
    test_df = df_shuffled[split_idx:]
    
    logger.info(f"Train set: {len(train_df)} records")
    logger.info(f"Test set: {len(test_df)} records")
    
    return train_df, test_df


def evaluate_recommendation_quality(model, model_name: str, 
                                   test_df: pd.DataFrame,
                                   evaluator: RecommenderEvaluator,
                                   n_samples: int = 50) -> dict:
    """
    Evaluate recommendation quality
    
    Args:
        model: Trained recommender model
        model_name: Name of the model
        test_df: Test dataset
        evaluator: Evaluator instance
        n_samples: Number of test samples
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Sample test books
    test_books = test_df.sample(min(n_samples, len(test_df)))['book_name'].tolist()
    
    precisions = []
    recalls = []
    f1_scores = []
    ndcgs = []
    
    for book in test_books:
        try:
            # Get recommendations
            recs = model.get_recommendations(book, top_n=10)
            
            if recs.empty:
                continue
            
            recommended_items = recs['book_name'].tolist()
            
            # Get ground truth (books with similar ratings in same genre)
            book_data = test_df[test_df['book_name'] == book]
            if book_data.empty:
                continue
            
            book_genre = book_data.iloc[0].get('genre', None)
            book_rating = book_data.iloc[0].get('rating', 0)
            
            # Find relevant items (similar genre and rating within 0.5)
            if book_genre and 'genre' in test_df.columns and 'rating' in test_df.columns:
                relevant_items = test_df[
                    (test_df['genre'] == book_genre) &
                    (test_df['rating'] >= book_rating - 0.5) &
                    (test_df['rating'] <= book_rating + 0.5) &
                    (test_df['book_name'] != book)
                ]['book_name'].tolist()
                
                if len(relevant_items) > 0:
                    # Calculate metrics
                    precision = evaluator.calculate_precision_at_k(recommended_items, relevant_items, 10)
                    recall = evaluator.calculate_recall_at_k(recommended_items, relevant_items, 10)
                    f1 = evaluator.calculate_f1_at_k(recommended_items, relevant_items, 10)
                    ndcg = evaluator.calculate_ndcg_at_k(recommended_items, relevant_items, 10)
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                    ndcgs.append(ndcg)
        
        except Exception as e:
            logger.debug(f"Error evaluating book '{book}': {e}")
            continue
    
    # Calculate averages
    avg_metrics = {
        'model_name': model_name,
        'precision@10': np.mean(precisions) if precisions else 0.0,
        'recall@10': np.mean(recalls) if recalls else 0.0,
        'f1@10': np.mean(f1_scores) if f1_scores else 0.0,
        'ndcg@10': np.mean(ndcgs) if ndcgs else 0.0,
        'num_evaluations': len(precisions)
    }
    
    return avg_metrics


def main():
    """Main evaluation execution"""
    
    print("\n" + "="*70)
    print("MODEL EVALUATION - RECOMMENDATION SYSTEMS")
    print("="*70 + "\n")
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator()
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv('data/processed/clustered_data.csv')
    
    # Create train/test split
    train_df, test_df = create_test_set(df, test_size=0.2)
    
    # Load and evaluate models
    models_path = Path("data/models")
    
    models_to_evaluate = [
        ('TF-IDF Recommender', TFIDFRecommender()),
        ('Content-Based Recommender', ContentBasedRecommender()),
        ('Clustering Recommender', ClusteringRecommender()),
        ('Hybrid Recommender', HybridRecommender())
    ]
    
    all_results = {}
    
    for model_name, model in models_to_evaluate:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Train model on training data
            logger.info(f"Training {model_name} on training set...")
            model.fit(train_df)
            
            # Evaluate
            metrics = evaluate_recommendation_quality(
                model, model_name, test_df, evaluator, n_samples=50
            )
            
            all_results[model_name] = metrics
            
            # Store in evaluator
            evaluator.evaluation_results[model_name] = metrics
            
            # Print results
            print(f"\nResults for {model_name}:")
            print(f"  Precision@10: {metrics['precision@10']:.4f}")
            print(f"  Recall@10:    {metrics['recall@10']:.4f}")
            print(f"  F1@10:        {metrics['f1@10']:.4f}")
            print(f"  NDCG@10:      {metrics['ndcg@10']:.4f}")
            print(f"  Evaluations:  {metrics['num_evaluations']}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    # Compare all models
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    comparison_df = evaluator.compare_models(save_plot=True)
    print("\nComparison Results:")
    print(comparison_df)
    
    # Generate reports
    print(f"\n{'='*70}")
    print("GENERATING REPORTS")
    print(f"{'='*70}\n")
    
    evaluator.plot_metric_heatmap(save_plot=True)
    evaluator.generate_evaluation_report()
    evaluator.export_results_to_csv()
    
    # Best model summary
    print(f"\n{'='*70}")
    print("BEST PERFORMING MODELS")
    print(f"{'='*70}\n")
    
    if not comparison_df.empty:
        for metric in ['precision@10', 'recall@10', 'f1@10', 'ndcg@10']:
            if metric in comparison_df.columns:
                best_idx = comparison_df[metric].idxmax()
                best_value = comparison_df[metric].max()
                print(f"{metric:15}: {best_idx:30} ({best_value:.4f})")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print("\nOutputs saved to:")
    print("  - outputs/visualizations/model_comparison.png")
    print("  - outputs/visualizations/metrics_heatmap.png")
    print("  - outputs/reports/model_evaluation_report.txt")
    print("  - outputs/reports/model_evaluation_results.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise
