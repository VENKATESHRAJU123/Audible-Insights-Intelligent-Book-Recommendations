"""
Unit Tests for Model Evaluation Module
Tests for evaluation metrics and model comparison
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import RecommenderEvaluator


class TestRecommenderEvaluator:
    """Test suite for RecommenderEvaluator"""
    
    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator instance"""
        return RecommenderEvaluator(output_path=str(tmp_path / "reports"))
    
    @pytest.fixture
    def sample_ratings(self):
        """Create sample actual and predicted ratings"""
        np.random.seed(42)
        actual = np.random.uniform(1, 5, 100)
        predicted = actual + np.random.normal(0, 0.5, 100)
        return actual, predicted
    
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.output_path.exists()
        assert evaluator.evaluation_results == {}
    
    
    def test_calculate_rmse(self, evaluator, sample_ratings):
        """Test RMSE calculation"""
        actual, predicted = sample_ratings
        rmse = evaluator.calculate_rmse(actual, predicted)
        
        assert rmse >= 0
        assert isinstance(rmse, float)
    
    
    def test_calculate_mae(self, evaluator, sample_ratings):
        """Test MAE calculation"""
        actual, predicted = sample_ratings
        mae = evaluator.calculate_mae(actual, predicted)
        
        assert mae >= 0
        assert isinstance(mae, float)
    
    
    def test_calculate_precision_at_k(self, evaluator):
        """Test Precision@K calculation"""
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        relevant = [2, 4, 6, 12, 14]
        
        precision = evaluator.calculate_precision_at_k(recommended, relevant, k=10)
        
        assert precision == 0.3
    
    
    def test_calculate_recall_at_k(self, evaluator):
        """Test Recall@K calculation"""
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        relevant = [2, 4, 6, 12, 14]
        
        recall = evaluator.calculate_recall_at_k(recommended, relevant, k=10)
        
        assert recall == 0.6
    
    
    def test_calculate_f1_at_k(self, evaluator):
        """Test F1@K calculation"""
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        relevant = [2, 4, 6, 12, 14]
        
        f1 = evaluator.calculate_f1_at_k(recommended, relevant, k=10)
        
        precision = 0.3
        recall = 0.6
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        assert abs(f1 - expected_f1) < 0.001
    
    
    def test_calculate_ndcg_at_k(self, evaluator):
        """Test NDCG@K calculation"""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5]
        
        ndcg = evaluator.calculate_ndcg_at_k(recommended, relevant, k=5)
        
        assert 0 <= ndcg <= 1
        assert isinstance(ndcg, float)
    
    
    def test_calculate_ndcg_perfect_ranking(self, evaluator):
        """Test NDCG with perfect ranking"""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3, 4, 5]
        
        ndcg = evaluator.calculate_ndcg_at_k(recommended, relevant, k=5)
        
        assert abs(ndcg - 1.0) < 0.001
    
    
    def test_calculate_map_at_k(self, evaluator):
        """Test MAP@K calculation"""
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        relevant = [2, 4, 6]
        
        map_score = evaluator.calculate_map_at_k(recommended, relevant, k=10)
        
        assert 0 <= map_score <= 1
        assert isinstance(map_score, float)
    
    
    def test_calculate_coverage(self, evaluator):
        """Test catalog coverage calculation"""
        recommendations_all = [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]
        ]
        catalog_size = 20
        
        coverage = evaluator.calculate_coverage(recommendations_all, catalog_size)
        
        expected_coverage = 7 / 20
        assert abs(coverage - expected_coverage) < 0.001
    
    
    def test_calculate_diversity(self, evaluator):
        """Test diversity calculation"""
        recommendations = [1, 2, 3]
        similarity_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        item_to_idx = {1: 0, 2: 1, 3: 2}
        
        diversity = evaluator.calculate_diversity(
            recommendations, similarity_matrix, item_to_idx
        )
        
        assert 0 <= diversity <= 1
        assert isinstance(diversity, float)
    
    
    def test_calculate_novelty(self, evaluator):
        """Test novelty calculation"""
        recommendations = [1, 2, 3]
        item_popularity = {1: 100, 2: 50, 3: 10}
        
        novelty = evaluator.calculate_novelty(recommendations, item_popularity)
        
        assert novelty >= 0
        assert isinstance(novelty, float)
    
    
    def test_evaluate_rating_predictions(self, evaluator, sample_ratings):
        """Test rating prediction evaluation"""
        actual, predicted = sample_ratings
        
        metrics = evaluator.evaluate_rating_predictions(
            "Test Model",
            actual,
            predicted
        )
        
        assert 'model_name' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r_squared' in metrics
        assert metrics['model_name'] == "Test Model"
    
    
    def test_evaluate_model(self, evaluator):
        """Test complete model evaluation"""
        recommendations_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'recommended_items': [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7]
            ]
        })
        
        ground_truth_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'relevant_items': [
                [1, 2, 10],
                [2, 4, 11],
                [3, 5, 12]
            ]
        })
        
        metrics = evaluator.evaluate_model(
            "Test Model",
            recommendations_df,
            ground_truth_df,
            k=5
        )
        
        assert 'model_name' in metrics
        assert 'precision@10' in metrics
        assert 'recall@10' in metrics
        assert 'f1@10' in metrics
    
    
    def test_compare_models(self, evaluator):
        """Test model comparison"""
        evaluator.evaluation_results = {
            'Model A': {
                'model_name': 'Model A',
                'precision@10': 0.75,
                'recall@10': 0.65,
                'f1@10': 0.70
            },
            'Model B': {
                'model_name': 'Model B',
                'precision@10': 0.80,
                'recall@10': 0.60,
                'f1@10': 0.69
            }
        }
        
        comparison_df = evaluator.compare_models(save_plot=False)
        
        assert not comparison_df.empty
        assert len(comparison_df) == 2
    
    
    def test_generate_evaluation_report(self, evaluator, tmp_path):
        """Test report generation"""
        evaluator.evaluation_results = {
            'Model A': {
                'model_name': 'Model A',
                'precision@10': 0.75,
                'recall@10': 0.65
            }
        }
        
        evaluator.generate_evaluation_report()
        
        report_file = evaluator.output_path / "model_evaluation_report.txt"
        assert report_file.exists()
    
    
    def test_export_results_to_csv(self, evaluator):
        """Test CSV export"""
        evaluator.evaluation_results = {
            'Model A': {
                'model_name': 'Model A',
                'precision@10': 0.75,
                'recall@10': 0.65
            }
        }
        
        evaluator.export_results_to_csv()
        
        csv_file = evaluator.output_path / "model_evaluation_results.csv"
        assert csv_file.exists()


class TestEvaluationMetrics:
    """Test specific evaluation metric calculations"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return RecommenderEvaluator()
    
    
    def test_precision_empty_recommendations(self, evaluator):
        """Test precision with empty recommendations"""
        precision = evaluator.calculate_precision_at_k([], [1, 2, 3], k=5)
        assert precision == 0.0
    
    
    def test_recall_empty_relevant(self, evaluator):
        """Test recall with empty relevant items"""
        recall = evaluator.calculate_recall_at_k([1, 2, 3], [], k=5)
        assert recall == 0.0
    
    
    def test_f1_zero_division(self, evaluator):
        """Test F1 score with zero precision and recall"""
        f1 = evaluator.calculate_f1_at_k([1, 2, 3], [4, 5, 6], k=3)
        assert f1 == 0.0
    
    
    def test_ndcg_no_relevant(self, evaluator):
        """Test NDCG with no relevant items"""
        ndcg = evaluator.calculate_ndcg_at_k([1, 2, 3], [], k=3)
        assert ndcg == 0.0
    
    
    def test_map_no_relevant(self, evaluator):
        """Test MAP with no relevant items"""
        map_score = evaluator.calculate_map_at_k([1, 2, 3], [], k=3)
        assert map_score == 0.0


def test_imports():
    """Test that evaluation module can be imported"""
    from src.evaluation import RecommenderEvaluator
    assert RecommenderEvaluator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
