"""
NLP Feature Extraction Module
Extracts features from text data for recommendation models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import ssl
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# Handle SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading punkt_tab...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from book data for ML models"""
    
    def __init__(self, processed_data_path: str = "data/processed"):
        """Initialize FeatureExtractor"""
        self.processed_data_path = Path(processed_data_path)
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        logger.info("FeatureExtractor initialized")
    
    
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load the cleaned dataset"""
        logger.info("Loading cleaned data...")
        
        file_path = self.processed_data_path / "cleaned_data.csv"
        df = pd.read_csv(file_path, encoding='utf-8')
        
        logger.info(f"Cleaned data loaded: {df.shape}")
        return df
    
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for NLP
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Join back
        return ' '.join(tokens)
    
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract TF-IDF features from text columns
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with text features
        """
        logger.info("Extracting text features...")
        
        df_features = df.copy()
        
        # Combine text columns for comprehensive feature extraction
        text_cols = []
        if 'description' in df.columns:
            text_cols.append('description')
        if 'genre' in df.columns:
            text_cols.append('genre')
        if 'book_name' in df.columns:
            text_cols.append('book_name')
        
        # Create combined text column
        df_features['combined_text'] = ''
        for col in text_cols:
            if col in df.columns:
                df_features['combined_text'] += ' ' + df_features[col].astype(str)
        
        # Preprocess text
        df_features['processed_text'] = df_features['combined_text'].apply(
            self.preprocess_text
        )
        
        # Extract TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            df_features['processed_text']
        )
        
        # Convert to dataframe
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        logger.info(f"TF-IDF features extracted: {tfidf_df.shape[1]} features")
        
        return pd.concat([df_features, tfidf_df], axis=1)
    
    
    def extract_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        df_features = df.copy()
        
        categorical_cols = ['author', 'genre']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df_features[f'{col}_encoded'] = le.fit_transform(
                    df_features[col].astype(str)
                )
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return df_features
    
    
    def extract_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with scaled numerical features
        """
        logger.info("Scaling numerical features...")
        
        df_features = df.copy()
        
        numerical_cols = []
        for col in ['rating', 'price']:
            if col in df.columns:
                numerical_cols.append(col)
        
        # Add review count columns
        review_cols = [col for col in df.columns if 'review' in col.lower()]
        numerical_cols.extend(review_cols)
        
        if numerical_cols:
            scaled_values = self.scaler.fit_transform(
                df_features[numerical_cols].fillna(0)
            )
            
            scaled_df = pd.DataFrame(
                scaled_values,
                columns=[f'{col}_scaled' for col in numerical_cols]
            )
            
            df_features = pd.concat([df_features, scaled_df], axis=1)
            logger.info(f"Scaled {len(numerical_cols)} numerical features")
        
        return df_features
    
    
    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature matrix
        
        Args:
            df: Input dataframe
            
        Returns:
            Complete feature matrix
        """
        logger.info("Creating feature matrix...")
        
        # Extract all features
        df_with_text = self.extract_text_features(df)
        df_with_categorical = self.extract_categorical_features(df_with_text)
        df_with_numerical = self.extract_numerical_features(df_with_categorical)
        
        # Select feature columns
        feature_cols = (
            [col for col in df_with_numerical.columns if col.startswith('tfidf_')] +
            [col for col in df_with_numerical.columns if col.endswith('_encoded')] +
            [col for col in df_with_numerical.columns if col.endswith('_scaled')]
        )
        
        # Keep original columns + features
        all_cols = list(df.columns) + feature_cols
        feature_matrix = df_with_numerical[all_cols]
        
        # Save feature matrix
        output_path = self.processed_data_path / "feature_matrix.csv"
        feature_matrix.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Feature matrix saved: {feature_matrix.shape}")
        logger.info(f"Saved to: {output_path}")
        
        return feature_matrix


def main():
    """Main execution function"""
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Load cleaned data
    df_cleaned = extractor.load_cleaned_data()
    
    # Create feature matrix
    feature_matrix = extractor.create_feature_matrix(df_cleaned)
    
    print("\n" + "="*50)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total features: {feature_matrix.shape[1]}")
    print(f"Total records: {feature_matrix.shape[0]}")
    
    logger.info("\n✅ Feature extraction completed successfully!")


if __name__ == "__main__":
    main()
