"""
Model Evaluation Module
Comprehensive evaluation metrics and comparison for recommendation models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    """Comprehensive evaluation framework for recommendation systems"""
    
    def __init__(self, output_path: str = "outputs/reports"):
        """
        Initialize Recommender Evaluator
        
        Args:
            output_path: Path to save evaluation reports
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_results = {}
        
        logger.info("RecommenderEvaluator initialized")
    
    
    def calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error
        
        Args:
            actual: Actual ratings
            predicted: Predicted ratings
            
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(actual, predicted))
    
    
    def calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            actual: Actual ratings
            predicted: Predicted ratings
            
        Returns:
            MAE value
        """
        return mean_absolute_error(actual, predicted)
    
    
    def calculate_precision_at_k(self, recommended_items: List, 
                                 relevant_items: List, k: int = 10) -> float:
        """
        Calculate Precision@K
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K value
        """
        if k <= 0 or len(recommended_items) == 0:
            return 0.0
        
        recommended_at_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        recommended_set = set(recommended_at_k)
        
        hits = len(relevant_set.intersection(recommended_set))
        
        return hits / k
    
    
    def calculate_recall_at_k(self, recommended_items: List, 
                              relevant_items: List, k: int = 10) -> float:
        """
        Calculate Recall@K
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K value
        """
        if len(relevant_items) == 0 or len(recommended_items) == 0:
            return 0.0
        
        recommended_at_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        recommended_set = set(recommended_at_k)
        
        hits = len(relevant_set.intersection(recommended_set))
        
        return hits / len(relevant_items)
    
    
    def calculate_f1_at_k(self, recommended_items: List, 
                         relevant_items: List, k: int = 10) -> float:
        """
        Calculate F1-Score@K
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            F1@K value
        """
        precision = self.calculate_precision_at_k(recommended_items, relevant_items, k)
        recall = self.calculate_recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    
    def calculate_ndcg_at_k(self, recommended_items: List, 
                           relevant_items: List, k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K value
        """
        def dcg_at_k(relevance_scores, k):
            """Calculate DCG@K"""
            relevance_scores = np.array(relevance_scores)[:k]
            if relevance_scores.size == 0:
                return 0.0
            
            # DCG = sum(rel_i / log2(i+2)) for i in [0, k)
            discounts = np.log2(np.arange(2, relevance_scores.size + 2))
            return np.sum(relevance_scores / discounts)
        
        # Create relevance scores (1 if relevant, 0 if not)
        recommended_at_k = recommended_items[:k]
        relevance_scores = [1 if item in relevant_items else 0 for item in recommended_at_k]
        
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal DCG - all relevant items at top)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    
    def calculate_map_at_k(self, recommended_items: List, 
                          relevant_items: List, k: int = 10) -> float:
        """
        Calculate Mean Average Precision (MAP@K)
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K value
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_at_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended_at_k, 1):
            if item in relevant_set:
                hits += 1
                precision_at_i = hits / i
                sum_precisions += precision_at_i
        
        if hits == 0:
            return 0.0
        
        return sum_precisions / min(len(relevant_items), k)
    
    
    def calculate_coverage(self, recommended_items_all: List[List], 
                          catalog_size: int) -> float:
        """
        Calculate catalog coverage
        
        Args:
            recommended_items_all: List of recommendation lists
            catalog_size: Total number of items in catalog
            
        Returns:
            Coverage percentage
        """
        unique_recommended = set()
        for recommendations in recommended_items_all:
            unique_recommended.update(recommendations)
        
        return len(unique_recommended) / catalog_size
    
    
    def calculate_diversity(self, recommendations: List, 
                           similarity_matrix: np.ndarray,
                           item_to_idx: Dict) -> float:
        """
        Calculate diversity of recommendations (1 - average similarity)
        
        Args:
            recommendations: List of recommended item IDs
            similarity_matrix: Item-item similarity matrix
            item_to_idx: Mapping from item ID to matrix index
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(recommendations) < 2:
            return 0.0
        
        similarities = []
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item_i = recommendations[i]
                item_j = recommendations[j]
                
                if item_i in item_to_idx and item_j in item_to_idx:
                    idx_i = item_to_idx[item_i]
                    idx_j = item_to_idx[item_j]
                    sim = similarity_matrix[idx_i, idx_j]
                    similarities.append(sim)
        
        if len(similarities) == 0:
            return 0.0
        
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity
        
        return diversity
    
    
    def calculate_novelty(self, recommendations: List, 
                         item_popularity: Dict) -> float:
        """
        Calculate novelty of recommendations (preference for less popular items)
        
        Args:
            recommendations: List of recommended item IDs
            item_popularity: Dictionary mapping item ID to popularity count
            
        Returns:
            Novelty score (higher is more novel)
        """
        if len(recommendations) == 0:
            return 0.0
        
        total_popularity = sum(item_popularity.values())
        novelty_scores = []
        
        for item in recommendations:
            if item in item_popularity:
                popularity = item_popularity[item]
                # Novelty = -log(popularity / total_popularity)
                if popularity > 0:
                    prob = popularity / total_popularity
                    novelty = -np.log2(prob)
                    novelty_scores.append(novelty)
        
        if len(novelty_scores) == 0:
            return 0.0
        
        return np.mean(novelty_scores)
    
    
    def evaluate_model(self, model_name: str, 
                      recommendations_df: pd.DataFrame,
                      ground_truth_df: pd.DataFrame,
                      k: int = 10) -> Dict:
        """
        Evaluate a recommendation model comprehensively
        
        Args:
            model_name: Name of the model
            recommendations_df: DataFrame with columns ['user_id', 'recommended_items']
            ground_truth_df: DataFrame with columns ['user_id', 'relevant_items']
            k: Number of top recommendations to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = {
            'model_name': model_name,
            'precision_at_k': [],
            'recall_at_k': [],
            'f1_at_k': [],
            'ndcg_at_k': [],
            'map_at_k': []
        }
        
        # Merge recommendations with ground truth
        merged = pd.merge(recommendations_df, ground_truth_df, on='user_id', how='inner')
        
        # Calculate metrics for each user
        for _, row in merged.iterrows():
            recommended = row['recommended_items']
            relevant = row['relevant_items']
            
            metrics['precision_at_k'].append(
                self.calculate_precision_at_k(recommended, relevant, k)
            )
            metrics['recall_at_k'].append(
                self.calculate_recall_at_k(recommended, relevant, k)
            )
            metrics['f1_at_k'].append(
                self.calculate_f1_at_k(recommended, relevant, k)
            )
            metrics['ndcg_at_k'].append(
                self.calculate_ndcg_at_k(recommended, relevant, k)
            )
            metrics['map_at_k'].append(
                self.calculate_map_at_k(recommended, relevant, k)
            )
        
        # Calculate average metrics
        avg_metrics = {
            'model_name': model_name,
            'precision@10': np.mean(metrics['precision_at_k']),
            'recall@10': np.mean(metrics['recall_at_k']),
            'f1@10': np.mean(metrics['f1_at_k']),
            'ndcg@10': np.mean(metrics['ndcg_at_k']),
            'map@10': np.mean(metrics['map_at_k']),
            'num_users_evaluated': len(merged)
        }
        
        self.evaluation_results[model_name] = avg_metrics
        
        logger.info(f"Evaluation complete for {model_name}")
        return avg_metrics
    
    
    def evaluate_rating_predictions(self, model_name: str,
                                   actual_ratings: np.ndarray,
                                   predicted_ratings: np.ndarray) -> Dict:
        """
        Evaluate rating prediction accuracy
        
        Args:
            model_name: Name of the model
            actual_ratings: Array of actual ratings
            predicted_ratings: Array of predicted ratings
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating rating predictions for: {model_name}")
        
        metrics = {
            'model_name': model_name,
            'rmse': self.calculate_rmse(actual_ratings, predicted_ratings),
            'mae': self.calculate_mae(actual_ratings, predicted_ratings),
            'num_predictions': len(actual_ratings)
        }
        
        # Calculate R-squared
        ss_res = np.sum((actual_ratings - predicted_ratings) ** 2)
        ss_tot = np.sum((actual_ratings - np.mean(actual_ratings)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics['r_squared'] = r_squared
        
        self.evaluation_results[f"{model_name}_ratings"] = metrics
        
        logger.info(f"Rating prediction evaluation complete for {model_name}")
        return metrics
    
    
    def compare_models(self, save_plot: bool = True) -> pd.DataFrame:
        """
        Compare all evaluated models
        
        Args:
            save_plot: Whether to save comparison plots
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing models...")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.evaluation_results).T
        
        # Select only ranking metrics (exclude rating prediction metrics)
        ranking_metrics = ['precision@10', 'recall@10', 'f1@10', 'ndcg@10', 'map@10']
        available_metrics = [m for m in ranking_metrics if m in comparison_df.columns]
        
        if not available_metrics:
            logger.warning("No ranking metrics available for comparison")
            return comparison_df
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        colors = sns.color_palette("husl", len(comparison_df))
        
        for idx, metric in enumerate(available_metrics):
            if idx < len(axes):
                ax = axes[idx]
                
                values = comparison_df[metric].values
                models = comparison_df['model_name'].values if 'model_name' in comparison_df.columns else comparison_df.index
                
                bars = ax.barh(range(len(values)), values, color=colors, edgecolor='black', alpha=0.7)
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels(models, fontsize=10)
                ax.set_xlabel('Score', fontsize=11)
                ax.set_title(f'{metric.upper()}', fontsize=13, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                ax.set_xlim(0, 1.0)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
        
        # Hide extra subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Model Comparison - Recommendation Metrics', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path.parent / 'visualizations' / 'model_comparison.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to: {output_file}")
        
        plt.show()
        
        return comparison_df
    
    
    def plot_precision_recall_curve(self, model_results: Dict[str, List[Tuple]], 
                                    save_plot: bool = True):
        """
        Plot precision-recall curves for multiple models
        
        Args:
            model_results: Dict mapping model names to list of (precision, recall) tuples
            save_plot: Whether to save the plot
        """
        logger.info("Plotting precision-recall curves...")
        
        plt.figure(figsize=(10, 8))
        
        colors = sns.color_palette("husl", len(model_results))
        
        for (model_name, results), color in zip(model_results.items(), colors):
            precisions = [r[0] for r in results]
            recalls = [r[1] for r in results]
            
            plt.plot(recalls, precisions, marker='o', label=model_name, 
                    color=color, linewidth=2, markersize=6)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        
        if save_plot:
            output_file = self.output_path.parent / 'visualizations' / 'precision_recall_curves.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curves saved to: {output_file}")
        
        plt.show()
    
    
    def plot_metric_heatmap(self, save_plot: bool = True):
        """
        Create heatmap of all metrics for all models
        
        Args:
            save_plot: Whether to save the plot
        """
        logger.info("Creating metric heatmap...")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return
        
        # Create dataframe
        comparison_df = pd.DataFrame(self.evaluation_results).T
        
        # Select numeric columns
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        heatmap_data = comparison_df[numeric_cols]
        
        # Create heatmap
        plt.figure(figsize=(12, max(6, len(heatmap_data) * 0.5)))
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.tight_layout()
        
        if save_plot:
            output_file = self.output_path.parent / 'visualizations' / 'metrics_heatmap.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics heatmap saved to: {output_file}")
        
        plt.show()
    
    
    def generate_evaluation_report(self, output_file: str = "model_evaluation_report.txt"):
        """
        Generate comprehensive text evaluation report
        
        Args:
            output_file: Filename for the report
        """
        logger.info("Generating evaluation report...")
        
        filepath = self.output_path / output_file
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Models Evaluated: {len(self.evaluation_results)}\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Individual model results
            for model_name, metrics in self.evaluation_results.items():
                f.write("-"*80 + "\n")
                f.write(f"MODEL: {model_name}\n")
                f.write("-"*80 + "\n")
                
                for metric_name, value in metrics.items():
                    if metric_name != 'model_name':
                        if isinstance(value, (int, float)):
                            f.write(f"{metric_name:25}: {value:.4f}\n")
                        else:
                            f.write(f"{metric_name:25}: {value}\n")
                f.write("\n")
            
            # Best performing models
            if self.evaluation_results:
                f.write("="*80 + "\n")
                f.write("BEST PERFORMING MODELS BY METRIC\n")
                f.write("="*80 + "\n\n")
                
                comparison_df = pd.DataFrame(self.evaluation_results).T
                
                for metric in ['precision@10', 'recall@10', 'f1@10', 'ndcg@10', 'map@10']:
                    if metric in comparison_df.columns:
                        best_model = comparison_df[metric].idxmax()
                        best_value = comparison_df[metric].max()
                        f.write(f"{metric:15} - Best: {best_model:30} ({best_value:.4f})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Evaluation report saved to: {filepath}")
        print(f"\n✅ Evaluation report saved to: {filepath}")
    
    
    def export_results_to_csv(self, output_file: str = "model_evaluation_results.csv"):
        """
        Export evaluation results to CSV
        
        Args:
            output_file: Filename for the CSV
        """
        logger.info("Exporting results to CSV...")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results to export")
            return
        
        filepath = self.output_path / output_file
        
        comparison_df = pd.DataFrame(self.evaluation_results).T
        comparison_df.to_csv(filepath, index=True)
        
        logger.info(f"Results exported to: {filepath}")
        print(f"\n✅ Results exported to: {filepath}")


def main():
    """Main execution function for testing"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION FRAMEWORK - DEMO")
    print("="*60 + "\n")
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator()
    
    # Example: Create dummy data for demonstration
    np.random.seed(42)
    
    # Simulate actual vs predicted ratings
    actual_ratings = np.random.uniform(1, 5, 100)
    predicted_ratings_model1 = actual_ratings + np.random.normal(0, 0.5, 100)
    predicted_ratings_model2 = actual_ratings + np.random.normal(0, 0.8, 100)
    
    # Evaluate rating predictions
    print("📊 Evaluating Rating Predictions...")
    metrics1 = evaluator.evaluate_rating_predictions(
        "Content-Based Model",
        actual_ratings,
        predicted_ratings_model1
    )
    print(f"\nContent-Based Model:")
    print(f"  RMSE: {metrics1['rmse']:.4f}")
    print(f"  MAE: {metrics1['mae']:.4f}")
    print(f"  R²: {metrics1['r_squared']:.4f}")
    
    metrics2 = evaluator.evaluate_rating_predictions(
        "Hybrid Model",
        actual_ratings,
        predicted_ratings_model2
    )
    print(f"\nHybrid Model:")
    print(f"  RMSE: {metrics2['rmse']:.4f}")
    print(f"  MAE: {metrics2['mae']:.4f}")
    print(f"  R²: {metrics2['r_squared']:.4f}")
    
    # Example: Evaluate ranking metrics
    print("\n📈 Evaluating Ranking Metrics...")
    
    # Create dummy recommendation and ground truth data
    recommendations_df = pd.DataFrame({
        'user_id': [1, 2, 3],
        'recommended_items': [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ]
    })
    
    ground_truth_df = pd.DataFrame({
        'user_id': [1, 2, 3],
        'relevant_items': [
            [1, 2, 3, 15, 16],
            [2, 4, 6, 17, 18],
            [3, 5, 7, 19, 20]
        ]
    })
    
    ranking_metrics = evaluator.evaluate_model(
        "TF-IDF Model",
        recommendations_df,
        ground_truth_df,
        k=10
    )
    
    print(f"\nTF-IDF Model Ranking Metrics:")
    for metric, value in ranking_metrics.items():
        if metric != 'model_name':
            print(f"  {metric}: {value:.4f}")
    
    # Generate visualizations
    print("\n📊 Generating Visualizations...")
    evaluator.plot_metric_heatmap()
    
    # Generate reports
    print("\n📄 Generating Reports...")
    evaluator.generate_evaluation_report()
    evaluator.export_results_to_csv()
    
    print("\n" + "="*60)
    print("✅ EVALUATION DEMO COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
