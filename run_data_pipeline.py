"""
Master Data Processing Pipeline
Runs all data processing steps in sequence
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import DataProcessor
from src.nlp_features import FeatureExtractor
from src.clustering import BookClusterer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline():
    """Run the complete data processing pipeline"""
    
    print("\n" + "="*60)
    print("BOOK RECOMMENDATION SYSTEM - DATA PROCESSING PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Data Processing (Merge + Clean)
        print("\n📊 STEP 1: Data Processing (Merge + Clean)")
        print("-" * 60)
        processor = DataProcessor()
        df1, df2 = processor.load_datasets()
        merged_df = processor.merge_datasets(df1, df2)
        cleaned_df = processor.clean_data(merged_df)
        
        # Step 2: Feature Extraction
        print("\n🔧 STEP 2: Feature Extraction")
        print("-" * 60)
        extractor = FeatureExtractor()
        feature_matrix = extractor.create_feature_matrix(cleaned_df)
        
        # Step 3: Clustering
        print("\n🎯 STEP 3: Clustering")
        print("-" * 60)
        clusterer = BookClusterer()
        X, feature_cols = clusterer.prepare_clustering_features(feature_matrix)
        kmeans_labels = clusterer.apply_kmeans(X)
        clusterer.visualize_clusters(feature_matrix, kmeans_labels, method="kmeans")
        dbscan_labels = clusterer.apply_dbscan(X, eps=1.5, min_samples=5)
        df_clustered = clusterer.create_clustered_data(
            feature_matrix, kmeans_labels, dbscan_labels
        )
        
        # Final Summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nGenerated Files:")
        print(f"1. data/processed/merged_data.csv ({len(merged_df)} rows)")
        print(f"2. data/processed/cleaned_data.csv ({len(cleaned_df)} rows)")
        print(f"3. data/processed/feature_matrix.csv ({feature_matrix.shape[1]} columns)")
        print(f"4. data/processed/clustered_data.csv ({len(df_clustered)} rows)")
        print(f"\nVisualization files saved in: outputs/visualizations/")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_complete_pipeline()
