"""
Data Processing Module
Handles merging, cleaning, and preprocessing of Audible datasets
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main class for data processing operations"""
    
    def __init__(self, raw_data_path: str = "data/raw", 
                 processed_data_path: str = "data/processed"):
        """
        Initialize the DataProcessor
        
        Args:
            raw_data_path: Path to raw data directory
            processed_data_path: Path to save processed data
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Create directories if they don't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataProcessor initialized")
    
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the two Audible datasets
        
        Returns:
            Tuple of (dataset1, dataset2)
        """
        logger.info("Loading datasets...")
        
        try:
            # Load Dataset 1
            df1 = pd.read_csv(
                self.raw_data_path / "Audible_Catalog.csv",
                encoding='utf-8'
            )
            logger.info(f"Dataset 1 loaded: {df1.shape}")
            
            # Load Dataset 2
            df2 = pd.read_csv(
                self.raw_data_path / "Audible_Catalog_Advanced_Features.csv",
                encoding='utf-8'
            )
            logger.info(f"Dataset 2 loaded: {df2.shape}")
            
            return df1, df2
            
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    
    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the two datasets on common columns
        
        Args:
            df1: First dataset
            df2: Second dataset
            
        Returns:
            Merged dataframe
        """
        logger.info("Merging datasets...")
        
        # Identify common columns
        common_cols = list(set(df1.columns) & set(df2.columns))
        logger.info(f"Common columns: {common_cols}")
        
        # Standardize column names
        df1.columns = df1.columns.str.strip().str.lower().str.replace(' ', '_')
        df2.columns = df2.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Merge on book_name and author (most common identifiers)
        merge_keys = []
        if 'book_name' in df1.columns and 'book_name' in df2.columns:
            merge_keys.append('book_name')
        if 'author' in df1.columns and 'author' in df2.columns:
            merge_keys.append('author')
        
        if not merge_keys:
            logger.warning("No common merge keys found. Using outer join on all columns.")
            merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)
        else:
            logger.info(f"Merging on keys: {merge_keys}")
            merged_df = pd.merge(
                df1, df2, 
                on=merge_keys, 
                how='outer',
                suffixes=('_catalog', '_advanced')
            )
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        # Save merged data
        output_path = self.processed_data_path / "merged_data.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Merged data saved to: {output_path}")
        
        return merged_df
    
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the merged dataset
        
        Args:
            df: Merged dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        
        df_clean = df.copy()
        
        # 1. Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # 2. Handle missing values
        logger.info("\nMissing values before cleaning:")
        logger.info(f"\n{df_clean.isnull().sum()}")
        
        # Fill missing ratings with median
        if 'rating' in df_clean.columns:
            median_rating = df_clean['rating'].median()
            df_clean['rating'].fillna(median_rating, inplace=True)
        
        # Fill missing number_of_reviews with 0
        review_cols = [col for col in df_clean.columns if 'review' in col.lower()]
        for col in review_cols:
            df_clean[col].fillna(0, inplace=True)
        
        # Fill missing prices with median
        if 'price' in df_clean.columns:
            df_clean['price'] = self._clean_price(df_clean['price'])
            median_price = df_clean['price'].median()
            df_clean['price'].fillna(median_price, inplace=True)
        
        # Fill missing text fields with empty string
        text_cols = ['description', 'genre', 'book_name', 'author']
        for col in text_cols:
            if col in df_clean.columns:
                df_clean[col].fillna('Unknown', inplace=True)
        
        # 3. Standardize text columns
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_clean[col] = df_clean[col].str.strip()
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        # 4. Remove rows with critical missing data
        if 'book_name' in df_clean.columns:
            df_clean = df_clean[df_clean['book_name'] != 'Unknown']
        
        logger.info(f"\nCleaned data shape: {df_clean.shape}")
        logger.info(f"Missing values after cleaning:\n{df_clean.isnull().sum()}")
        
        # Save cleaned data
        output_path = self.processed_data_path / "cleaned_data.csv"
        df_clean.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Cleaned data saved to: {output_path}")
        
        return df_clean
    
    
    def _clean_price(self, price_series: pd.Series) -> pd.Series:
        """
        Clean price column (remove currency symbols, convert to float)
        
        Args:
            price_series: Price column
            
        Returns:
            Cleaned price series
        """
        def extract_price(price):
            if pd.isna(price):
                return np.nan
            # Remove currency symbols and convert to float
            price_str = str(price).replace('$', '').replace('£', '').replace('€', '')
            price_str = re.sub(r'[^\d.]', '', price_str)
            try:
                return float(price_str)
            except ValueError:
                return np.nan
        
        return price_series.apply(extract_price)
    
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> dict:
        """
        Generate a data quality report
        
        Args:
            df: Dataframe to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_rows': df.duplicated().sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_columns': df.select_dtypes(include=['object']).columns.tolist(),
        }
        
        return report


def main():
    """Main execution function"""
    
    # Initialize processor
    processor = DataProcessor()
    
    # Step 1: Load datasets
    df1, df2 = processor.load_datasets()
    
    # Step 2: Merge datasets
    merged_df = processor.merge_datasets(df1, df2)
    
    # Step 3: Clean data
    cleaned_df = processor.clean_data(merged_df)
    
    # Step 4: Generate quality report
    report = processor.generate_data_quality_report(cleaned_df)
    
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    print(f"Total Rows: {report['total_rows']}")
    print(f"Total Columns: {report['total_columns']}")
    print(f"Duplicate Rows: {report['duplicate_rows']}")
    print(f"\nNumeric Columns: {len(report['numeric_columns'])}")
    print(f"Text Columns: {len(report['text_columns'])}")
    
    logger.info("\n✅ Data processing completed successfully!")


if __name__ == "__main__":
    main()
