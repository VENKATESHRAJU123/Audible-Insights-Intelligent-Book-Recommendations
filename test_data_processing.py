"""
Unit Tests for Data Processing Module
Tests for data loading, merging, cleaning, and preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import DataProcessor


class TestDataProcessor:
    """Test suite for DataProcessor class"""
    
    @pytest.fixture
    def sample_df1(self):
        """Create sample dataset 1"""
        return pd.DataFrame({
            'Book Name': ['Book A', 'Book B', 'Book C', 'Book D', 'Book E'],
            'Author': ['Author 1', 'Author 2', 'Author 1', 'Author 3', 'Author 2'],
            'Rating': [4.5, 3.8, 4.2, None, 4.0],
            'Genre': ['Fiction', 'Mystery', 'Fiction', 'Romance', None],
            'Price': [15.99, 12.50, 18.00, None, 14.99]
        })
    
    @pytest.fixture
    def sample_df2(self):
        """Create sample dataset 2"""
        return pd.DataFrame({
            'Book Name': ['Book A', 'Book B', 'Book F', 'Book G'],
            'Author': ['Author 1', 'Author 2', 'Author 4', 'Author 5'],
            'Rating': [4.5, 3.8, 4.7, 3.5],
            'Number of Reviews': [100, 50, 200, 30],
            'Price': [15.99, 12.50, 20.00, 10.00]
        })
    
    @pytest.fixture
    def processor(self, tmp_path):
        """Create DataProcessor instance with temp paths"""
        return DataProcessor(
            raw_data_path=str(tmp_path / "raw"),
            processed_data_path=str(tmp_path / "processed")
        )
    
    
    def test_processor_initialization(self, processor):
        """Test DataProcessor initialization"""
        assert processor.raw_data_path.exists()
        assert processor.processed_data_path.exists()
    
    
    def test_merge_datasets(self, processor, sample_df1, sample_df2):
        """Test dataset merging"""
        merged = processor.merge_datasets(sample_df1, sample_df2)
        
        assert not merged.empty
        assert 'book_name' in merged.columns
        assert 'author' in merged.columns
        assert len(merged) >= max(len(sample_df1), len(sample_df2))
    
    
    def test_merge_datasets_with_common_keys(self, processor, sample_df1, sample_df2):
        """Test merging on common keys"""
        merged = processor.merge_datasets(sample_df1, sample_df2)
        
        # Check that Book A and Book B are in merged data
        assert 'Book A' in merged['book_name'].values
        assert 'Book B' in merged['book_name'].values
    
    
    def test_clean_data_removes_duplicates(self, processor):
        """Test duplicate removal"""
        df_with_duplicates = pd.DataFrame({
            'book_name': ['Book A', 'Book A', 'Book B'],
            'author': ['Author 1', 'Author 1', 'Author 2'],
            'rating': [4.5, 4.5, 3.8]
        })
        
        cleaned = processor.clean_data(df_with_duplicates)
        assert len(cleaned) == 2
    
    
    def test_clean_data_fills_missing_ratings(self, processor, sample_df1):
        """Test missing rating imputation"""
        cleaned = processor.clean_data(sample_df1)
        
        assert cleaned['rating'].isnull().sum() == 0
    
    
    def test_clean_data_fills_missing_genre(self, processor, sample_df1):
        """Test missing genre filling"""
        cleaned = processor.clean_data(sample_df1)
        
        assert cleaned['genre'].isnull().sum() == 0
        assert 'Unknown' in cleaned['genre'].values
    
    
    def test_clean_price(self, processor):
        """Test price cleaning"""
        price_series = pd.Series(['$15.99', '12.50', '$18', 'N/A', None])
        cleaned_prices = processor._clean_price(price_series)
        
        assert cleaned_prices[0] == 15.99
        assert cleaned_prices[1] == 12.50
        assert cleaned_prices[2] == 18.0
        assert pd.isna(cleaned_prices[3])
    
    
    def test_generate_data_quality_report(self, processor, sample_df1):
        """Test data quality report generation"""
        report = processor.generate_data_quality_report(sample_df1)
        
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert 'duplicate_rows' in report
        assert report['total_rows'] == len(sample_df1)
        assert report['total_columns'] == len(sample_df1.columns)
    
    
    def test_clean_data_removes_unknown_books(self, processor):
        """Test removal of books with missing names"""
        df = pd.DataFrame({
            'book_name': ['Book A', None, 'Unknown', 'Book B'],
            'author': ['Author 1', 'Author 2', 'Author 3', 'Author 4'],
            'rating': [4.5, 3.8, 4.2, 4.0]
        })
        
        cleaned = processor.clean_data(df)
        assert 'Unknown' not in cleaned['book_name'].values
        assert len(cleaned) < len(df)
    
    
    def test_clean_data_handles_empty_dataframe(self, processor):
        """Test cleaning of empty dataframe"""
        empty_df = pd.DataFrame()
        cleaned = processor.clean_data(empty_df)
        
        assert cleaned.empty


class TestDataProcessingIntegration:
    """Integration tests for data processing pipeline"""
    
    @pytest.fixture
    def create_test_files(self, tmp_path):
        """Create test CSV files"""
        raw_path = tmp_path / "raw"
        raw_path.mkdir()
        
        df1 = pd.DataFrame({
            'Book Name': ['Book A', 'Book B', 'Book C'],
            'Author': ['Author 1', 'Author 2', 'Author 1'],
            'Rating': [4.5, 3.8, 4.2],
            'Genre': ['Fiction', 'Mystery', 'Fiction']
        })
        
        df2 = pd.DataFrame({
            'Book Name': ['Book A', 'Book B', 'Book D'],
            'Author': ['Author 1', 'Author 2', 'Author 3'],
            'Rating': [4.5, 3.8, 4.0],
            'Price': [15.99, 12.50, 18.00]
        })
        
        df1.to_csv(raw_path / "Audible_Catalog.csv", index=False)
        df2.to_csv(raw_path / "Audible_Catalog_Advanced_Features.csv", index=False)
        
        return tmp_path
    
    
    def test_full_pipeline(self, create_test_files):
        """Test complete data processing pipeline"""
        processor = DataProcessor(
            raw_data_path=str(create_test_files / "raw"),
            processed_data_path=str(create_test_files / "processed")
        )
        
        # Load
        df1, df2 = processor.load_datasets()
        assert not df1.empty
        assert not df2.empty
        
        # Merge
        merged = processor.merge_datasets(df1, df2)
        assert not merged.empty
        
        # Clean
        cleaned = processor.clean_data(merged)
        assert not cleaned.empty
        
        # Check output files exist
        assert (create_test_files / "processed" / "merged_data.csv").exists()
        assert (create_test_files / "processed" / "cleaned_data.csv").exists()


def test_imports():
    """Test that all required modules can be imported"""
    from src.data_processing import DataProcessor
    assert DataProcessor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
