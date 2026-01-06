"""
Jupyter Notebook Generation Script
Creates all analysis notebooks with pre-filled content
"""

import nbformat as nbf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_notebook_01_data_exploration():
    """Create 01_data_exploration.ipynb"""
    
    nb = nbf.v4.new_notebook()
    
    nb.cells = [
        # Title
        nbf.v4.new_markdown_cell("""# 01 - Data Exploration
        
## Objective
- Load and explore the raw Audible datasets
- Understand the structure and basic statistics
- Identify data quality issues

---"""),
        
        # Setup
        nbf.v4.new_markdown_cell("## 1. Setup and Imports"),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Libraries imported successfully!")"""),
        
        # Load Data
        nbf.v4.new_markdown_cell("## 2. Load Datasets"),
        nbf.v4.new_code_cell("""# Load Dataset 1
df1 = pd.read_csv('../data/raw/Audible_Catalog.csv', encoding='utf-8')
print("Dataset 1 Shape:", df1.shape)
print("\\nDataset 1 Columns:")
print(df1.columns.tolist())

# Load Dataset 2
df2 = pd.read_csv('../data/raw/Audible_Catalog_Advanced_Features.csv', encoding='utf-8')
print("\\nDataset 2 Shape:", df2.shape)
print("\\nDataset 2 Columns:")
print(df2.columns.tolist())"""),
        
        # Dataset 1 Overview
        nbf.v4.new_markdown_cell("## 3. Dataset 1 - Overview"),
        nbf.v4.new_code_cell("""# First few rows
print("First 5 rows of Dataset 1:")
df1.head()"""),
        
        nbf.v4.new_code_cell("""# Data types and info
print("Dataset 1 - Data Types and Info:")
df1.info()"""),
        
        nbf.v4.new_code_cell("""# Basic statistics
print("Dataset 1 - Descriptive Statistics:")
df1.describe(include='all')"""),
        
        nbf.v4.new_code_cell("""# Missing values
print("Dataset 1 - Missing Values:")
missing_df1 = pd.DataFrame({
    'Column': df1.columns,
    'Missing_Count': df1.isnull().sum(),
    'Missing_Percentage': (df1.isnull().sum() / len(df1) * 100).round(2)
})
missing_df1 = missing_df1[missing_df1['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_df1)

# Visualize
if not missing_df1.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(missing_df1['Column'], missing_df1['Missing_Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Values - Dataset 1')
    plt.tight_layout()
    plt.show()"""),
        
        # Dataset 2 Overview
        nbf.v4.new_markdown_cell("## 4. Dataset 2 - Overview"),
        nbf.v4.new_code_cell("""# First few rows
print("First 5 rows of Dataset 2:")
df2.head()"""),
        
        nbf.v4.new_code_cell("""# Data types and info
print("Dataset 2 - Data Types and Info:")
df2.info()"""),
        
        nbf.v4.new_code_cell("""# Basic statistics
print("Dataset 2 - Descriptive Statistics:")
df2.describe(include='all')"""),
        
        nbf.v4.new_code_cell("""# Missing values
print("Dataset 2 - Missing Values:")
missing_df2 = pd.DataFrame({
    'Column': df2.columns,
    'Missing_Count': df2.isnull().sum(),
    'Missing_Percentage': (df2.isnull().sum() / len(df2) * 100).round(2)
})
missing_df2 = missing_df2[missing_df2['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_df2)

# Visualize
if not missing_df2.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(missing_df2['Column'], missing_df2['Missing_Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Values - Dataset 2')
    plt.tight_layout()
    plt.show()"""),
        
        # Compare Datasets
        nbf.v4.new_markdown_cell("## 5. Compare Datasets"),
        nbf.v4.new_code_cell("""# Find common columns
common_cols = list(set(df1.columns) & set(df2.columns))
print(f"Common Columns ({len(common_cols)}):")
print(common_cols)

# Unique columns
df1_unique = list(set(df1.columns) - set(df2.columns))
df2_unique = list(set(df2.columns) - set(df1.columns))

print(f"\\nColumns only in Dataset 1 ({len(df1_unique)}):")
print(df1_unique)

print(f"\\nColumns only in Dataset 2 ({len(df2_unique)}):")
print(df2_unique)"""),
        
        # Data Quality Issues
        nbf.v4.new_markdown_cell("## 6. Data Quality Assessment"),
        nbf.v4.new_code_cell("""# Check for duplicates
print("Dataset 1 - Duplicates:")
print(f"Total rows: {len(df1)}")
print(f"Duplicate rows: {df1.duplicated().sum()}")

print("\\nDataset 2 - Duplicates:")
print(f"Total rows: {len(df2)}")
print(f"Duplicate rows: {df2.duplicated().sum()}")"""),
        
        nbf.v4.new_code_cell("""# Check data types consistency
print("Data Type Analysis - Dataset 1:")
for col in df1.columns:
    dtype = df1[col].dtype
    unique_count = df1[col].nunique()
    print(f"{col:30} | Type: {str(dtype):10} | Unique: {unique_count}")"""),
        
        # Key Findings
        nbf.v4.new_markdown_cell("""## 7. Key Findings

**Dataset Sizes:**
- Dataset 1: [Will be filled after running]
- Dataset 2: [Will be filled after running]

**Common Columns:**
- [List common columns]

**Data Quality Issues:**
1. Missing values in: [List columns]
2. Duplicate records: [Count]
3. Inconsistent data types: [List issues]

**Next Steps:**
1. Merge datasets on common columns
2. Handle missing values
3. Remove duplicates
4. Standardize data types

---

**Notebook 01 Complete** ✅"""),
    ]
    
    return nb


def create_notebook_02_data_cleaning():
    """Create 02_data_cleaning.ipynb"""
    
    nb = nbf.v4.new_notebook()
    
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 02 - Data Cleaning

## Objective
- Merge the two datasets
- Handle missing values
- Remove duplicates
- Standardize data formats
- Create cleaned dataset

---"""),
        
        nbf.v4.new_markdown_cell("## 1. Setup and Imports"),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

print("✅ Libraries imported successfully!")"""),
        
        nbf.v4.new_markdown_cell("## 2. Load Raw Data"),
        nbf.v4.new_code_cell("""# Load datasets
df1 = pd.read_csv('../data/raw/Audible_Catalog.csv', encoding='utf-8')
df2 = pd.read_csv('../data/raw/Audible_Catalog_Advanced_Features.csv', encoding='utf-8')

print(f"Dataset 1: {df1.shape}")
print(f"Dataset 2: {df2.shape}")"""),
        
        nbf.v4.new_markdown_cell("## 3. Standardize Column Names"),
        nbf.v4.new_code_cell("""# Standardize column names
df1.columns = df1.columns.str.strip().str.lower().str.replace(' ', '_')
df2.columns = df2.columns.str.strip().str.lower().str.replace(' ', '_')

print("Standardized Dataset 1 columns:")
print(df1.columns.tolist())
print("\\nStandardized Dataset 2 columns:")
print(df2.columns.tolist())"""),
        
        nbf.v4.new_markdown_cell("## 4. Merge Datasets"),
        nbf.v4.new_code_cell("""# Identify merge keys
merge_keys = []
if 'book_name' in df1.columns and 'book_name' in df2.columns:
    merge_keys.append('book_name')
if 'author' in df1.columns and 'author' in df2.columns:
    merge_keys.append('author')

print(f"Merge keys: {merge_keys}")

# Merge datasets
if merge_keys:
    df_merged = pd.merge(df1, df2, on=merge_keys, how='outer', suffixes=('_catalog', '_advanced'))
else:
    df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)

print(f"\\nMerged dataset shape: {df_merged.shape}")
df_merged.head()"""),
        
        nbf.v4.new_markdown_cell("## 5. Handle Duplicate Columns"),
        nbf.v4.new_code_cell("""# Check for duplicate column patterns
duplicate_cols = [col for col in df_merged.columns if '_catalog' in col or '_advanced' in col]
print(f"Duplicate column patterns found: {len(duplicate_cols)}")
print(duplicate_cols[:10])  # Show first 10

# Merge duplicate columns
for col in df_merged.columns:
    if col.endswith('_catalog'):
        base_name = col.replace('_catalog', '')
        advanced_col = base_name + '_advanced'
        
        if advanced_col in df_merged.columns:
            # Combine columns (prefer non-null values)
            df_merged[base_name] = df_merged[col].combine_first(df_merged[advanced_col])
            df_merged.drop([col, advanced_col], axis=1, inplace=True)
            print(f"Merged: {col} + {advanced_col} -> {base_name}")

print(f"\\nDataset shape after merging duplicate columns: {df_merged.shape}")"""),
        
        nbf.v4.new_markdown_cell("## 6. Remove Duplicate Rows"),
        nbf.v4.new_code_cell("""# Check duplicates
print(f"Duplicate rows before: {df_merged.duplicated().sum()}")

# Remove duplicates
df_merged = df_merged.drop_duplicates()

print(f"Duplicate rows after: {df_merged.duplicated().sum()}")
print(f"Dataset shape: {df_merged.shape}")"""),
        
        nbf.v4.new_markdown_cell("## 7. Handle Missing Values"),
        nbf.v4.new_code_cell("""# Analyze missing values
missing_analysis = pd.DataFrame({
    'Column': df_merged.columns,
    'Missing_Count': df_merged.isnull().sum(),
    'Missing_Percentage': (df_merged.isnull().sum() / len(df_merged) * 100).round(2)
})
missing_analysis = missing_analysis[missing_analysis['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print("Missing Values Analysis:")
print(missing_analysis)

# Visualize
if not missing_analysis.empty:
    plt.figure(figsize=(12, 6))
    plt.barh(missing_analysis['Column'], missing_analysis['Missing_Percentage'])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Values Before Cleaning')
    plt.tight_layout()
    plt.show()"""),
        
        nbf.v4.new_code_cell("""# Handle missing values strategically

# 1. Fill missing ratings with median
if 'rating' in df_merged.columns:
    median_rating = df_merged['rating'].median()
    df_merged['rating'].fillna(median_rating, inplace=True)
    print(f"Filled missing ratings with median: {median_rating}")

# 2. Fill missing review counts with 0
review_cols = [col for col in df_merged.columns if 'review' in col.lower()]
for col in review_cols:
    df_merged[col].fillna(0, inplace=True)
print(f"Filled {len(review_cols)} review columns with 0")

# 3. Fill missing text fields with 'Unknown'
text_cols = ['description', 'genre', 'author']
for col in text_cols:
    if col in df_merged.columns:
        df_merged[col].fillna('Unknown', inplace=True)
print(f"Filled {len([c for c in text_cols if c in df_merged.columns])} text columns with 'Unknown'")

# 4. Fill missing prices with median
if 'price' in df_merged.columns:
    # Clean price column first
    df_merged['price'] = df_merged['price'].astype(str).str.replace('$', '').str.replace(',', '')
    df_merged['price'] = pd.to_numeric(df_merged['price'], errors='coerce')
    
    median_price = df_merged['price'].median()
    df_merged['price'].fillna(median_price, inplace=True)
    print(f"Filled missing prices with median: ${median_price:.2f}")

# Check remaining missing values
print(f"\\nRemaining missing values: {df_merged.isnull().sum().sum()}")"""),
        
        nbf.v4.new_markdown_cell("## 8. Data Type Conversions"),
        nbf.v4.new_code_cell("""# Convert data types
print("Converting data types...")

# Convert rating to float
if 'rating' in df_merged.columns:
    df_merged['rating'] = pd.to_numeric(df_merged['rating'], errors='coerce')

# Convert price to float
if 'price' in df_merged.columns:
    df_merged['price'] = pd.to_numeric(df_merged['price'], errors='coerce')

# Convert review counts to integer
for col in review_cols:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0).astype(int)

print("Data type conversions complete!")
print("\\nData types:")
print(df_merged.dtypes)"""),
        
        nbf.v4.new_markdown_cell("## 9. Text Cleaning"),
        nbf.v4.new_code_cell("""# Clean text columns
def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).strip()
    text = re.sub(r'\\s+', ' ', text)  # Remove extra whitespace
    return text

# Apply to text columns
text_columns = df_merged.select_dtypes(include=['object']).columns
for col in text_columns:
    df_merged[col] = df_merged[col].apply(clean_text)

print(f"Cleaned {len(text_columns)} text columns")"""),
        
        nbf.v4.new_markdown_cell("## 10. Remove Invalid Records"),
        nbf.v4.new_code_cell("""# Remove records with missing critical fields
initial_count = len(df_merged)

# Remove books without names
if 'book_name' in df_merged.columns:
    df_merged = df_merged[df_merged['book_name'].str.len() > 0]
    df_merged = df_merged[df_merged['book_name'] != 'Unknown']

# Remove invalid ratings
if 'rating' in df_merged.columns:
    df_merged = df_merged[(df_merged['rating'] >= 0) & (df_merged['rating'] <= 5)]

final_count = len(df_merged)
print(f"Removed {initial_count - final_count} invalid records")
print(f"Final dataset shape: {df_merged.shape}")"""),
        
        nbf.v4.new_markdown_cell("## 11. Save Cleaned Data"),
        nbf.v4.new_code_cell("""# Create output directory
Path('../data/processed').mkdir(parents=True, exist_ok=True)

# Save merged data
df_merged.to_csv('../data/processed/merged_data.csv', index=False, encoding='utf-8')
print("✅ Merged data saved to: data/processed/merged_data.csv")

# Save cleaned data
df_merged.to_csv('../data/processed/cleaned_data.csv', index=False, encoding='utf-8')
print("✅ Cleaned data saved to: data/processed/cleaned_data.csv")

print(f"\\nFinal dataset: {df_merged.shape}")"""),
        
        nbf.v4.new_markdown_cell("## 12. Data Quality Report"),
        nbf.v4.new_code_cell("""# Generate data quality report
print("="*60)
print("DATA QUALITY REPORT")
print("="*60)
print(f"Total Records: {len(df_merged)}")
print(f"Total Columns: {len(df_merged.columns)}")
print(f"Missing Values: {df_merged.isnull().sum().sum()}")
print(f"Duplicate Rows: {df_merged.duplicated().sum()}")
print(f"\\nData Types:")
print(df_merged.dtypes.value_counts())

# Sample records
print("\\nSample Records:")
df_merged.head(3)"""),
        
        nbf.v4.new_markdown_cell("""## Summary

**Data Cleaning Steps Completed:**
1. ✅ Standardized column names
2. ✅ Merged two datasets
3. ✅ Handled duplicate columns
4. ✅ Removed duplicate rows
5. ✅ Filled missing values
6. ✅ Converted data types
7. ✅ Cleaned text fields
8. ✅ Removed invalid records

**Output Files:**
- `merged_data.csv` - Combined dataset
- `cleaned_data.csv` - Fully cleaned dataset

**Next Steps:**
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model building

---

**Notebook 02 Complete** ✅"""),
    ]
    
    return nb


def create_notebook_03_eda_analysis():
    """Create 03_eda_analysis.ipynb"""
    
    nb = nbf.v4.new_notebook()
    
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 03 - Exploratory Data Analysis (EDA)

## Objective
- Analyze book characteristics and distributions
- Discover patterns and trends
- Answer key business questions
- Create insightful visualizations

---"""),
        
        nbf.v4.new_markdown_cell("## 1. Setup and Imports"),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("✅ Libraries imported successfully!")"""),
        
        nbf.v4.new_markdown_cell("## 2. Load Cleaned Data"),
        nbf.v4.new_code_cell("""# Load cleaned dataset
df = pd.read_csv('../data/processed/cleaned_data.csv', encoding='utf-8')

print(f"Dataset shape: {df.shape}")
print(f"\\nColumns: {df.columns.tolist()}")
print(f"\\nFirst few rows:")
df.head()"""),
        
        nbf.v4.new_code_cell("""# Basic statistics
print("Dataset Overview:")
print(f"Total Books: {len(df):,}")
print(f"Total Authors: {df['author'].nunique() if 'author' in df.columns else 'N/A'}")
print(f"Total Genres: {df['genre'].nunique() if 'genre' in df.columns else 'N/A'}")
print(f"\\nNumeric Summary:")
df.describe()"""),
        
        nbf.v4.new_markdown_cell("## 3. Rating Analysis"),
        nbf.v4.new_code_cell("""# Rating distribution
if 'rating' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(df['rating'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['rating'].mean(), color='red', linestyle='--', label=f'Mean: {df["rating"].mean():.2f}')
    axes[0].axvline(df['rating'].median(), color='green', linestyle='--', label=f'Median: {df["rating"].median():.2f}')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Rating Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df['rating'].dropna(), vert=True)
    axes[1].set_ylabel('Rating')
    axes[1].set_title('Rating Box Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print("Rating Statistics:")
    print(f"Mean: {df['rating'].mean():.2f}")
    print(f"Median: {df['rating'].median():.2f}")
    print(f"Std Dev: {df['rating'].std():.2f}")
    print(f"Min: {df['rating'].min():.2f}")
    print(f"Max: {df['rating'].max():.2f}")"""),
        
        nbf.v4.new_markdown_cell("## 4. Genre Analysis"),
        nbf.v4.new_code_cell("""# Top genres
if 'genre' in df.columns:
    genre_counts = df['genre'].value_counts().head(15)
    
    # Bar plot
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='barh')
    plt.xlabel('Number of Books')
    plt.title('Top 15 Genres')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Genres:")
    print(genre_counts.head(10))"""),
        
        nbf.v4.new_code_cell("""# Genre vs Rating
if 'genre' in df.columns and 'rating' in df.columns:
    top_genres = df['genre'].value_counts().head(10).index
    genre_ratings = df[df['genre'].isin(top_genres)].groupby('genre')['rating'].agg(['mean', 'count'])
    genre_ratings = genre_ratings.sort_values('mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(genre_ratings))
    ax.bar(x, genre_ratings['mean'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(genre_ratings.index, rotation=45, ha='right')
    ax.set_ylabel('Average Rating')
    ax.set_title('Average Rating by Top 10 Genres')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels
    for i, (idx, row) in enumerate(genre_ratings.iterrows()):
        ax.text(i, row['mean'] + 0.05, f"n={row['count']}", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("Genre Rating Statistics:")
    print(genre_ratings)"""),
        
        nbf.v4.new_markdown_cell("## 5. Author Analysis"),
        nbf.v4.new_code_cell("""# Top authors by number of books
if 'author' in df.columns:
    top_authors = df['author'].value_counts().head(15)
    
    plt.figure(figsize=(12, 6))
    top_authors.plot(kind='barh', color='steelblue')
    plt.xlabel('Number of Books')
    plt.title('Top 15 Most Prolific Authors')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Authors:")
    print(top_authors.head(10))"""),
        
        nbf.v4.new_code_cell("""# Top rated authors (with minimum 3 books)
if 'author' in df.columns and 'rating' in df.columns:
    author_stats = df.groupby('author').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    author_stats.columns = ['author', 'avg_rating', 'book_count']
    
    # Filter authors with at least 3 books
    top_rated_authors = author_stats[author_stats['book_count'] >= 3].sort_values('avg_rating', ascending=False).head(15)
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_rated_authors)), top_rated_authors['avg_rating'])
    plt.yticks(range(len(top_rated_authors)), top_rated_authors['author'])
    plt.xlabel('Average Rating')
    plt.title('Top 15 Highest Rated Authors (min 3 books)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Top Rated Authors:")
    print(top_rated_authors)"""),
        
        nbf.v4.new_markdown_cell("## 6. Price Analysis"),
        nbf.v4.new_code_cell("""# Price distribution
if 'price' in df.columns:
    # Remove outliers for better visualization
    price_clean = df['price'][df['price'] < df['price'].quantile(0.95)]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(price_clean, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0].axvline(df['price'].mean(), color='red', linestyle='--', label=f'Mean: ${df["price"].mean():.2f}')
    axes[0].axvline(df['price'].median(), color='blue', linestyle='--', label=f'Median: ${df["price"].median():.2f}')
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Price Distribution (95th percentile)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(price_clean, vert=True)
    axes[1].set_ylabel('Price ($)')
    axes[1].set_title('Price Box Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Price Statistics:")
    print(f"Mean: ${df['price'].mean():.2f}")
    print(f"Median: ${df['price'].median():.2f}")
    print(f"Std Dev: ${df['price'].std():.2f}")
    print(f"Min: ${df['price'].min():.2f}")
    print(f"Max: ${df['price'].max():.2f}")"""),
        
        nbf.v4.new_markdown_cell("## 7. Correlation Analysis"),
        nbf.v4.new_code_cell("""# Correlation matrix for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numeric Features')
plt.tight_layout()
plt.show()

print("Correlation Matrix:")
print(correlation_matrix)"""),
        
        nbf.v4.new_markdown_cell("## 8. Rating vs Review Count"),
        nbf.v4.new_code_cell("""# Scatter plot: Rating vs Number of Reviews
review_cols = [col for col in df.columns if 'review' in col.lower()]

if review_cols and 'rating' in df.columns:
    review_col = review_cols[0]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df[review_col], df['rating'], alpha=0.5, s=20)
    plt.xlabel(f'Number of Reviews ({review_col})')
    plt.ylabel('Rating')
    plt.title('Rating vs Number of Reviews')
    plt.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df[review_col].dropna(), df['rating'].dropna(), 1)
    p = np.poly1d(z)
    plt.plot(df[review_col], p(df[review_col]), "r--", alpha=0.8, label='Trend')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Correlation
    corr = df[[review_col, 'rating']].corr().iloc[0, 1]
    print(f"Correlation between {review_col} and rating: {corr:.3f}")"""),
        
        nbf.v4.new_markdown_cell("## 9. Interactive Visualizations (Plotly)"),
        nbf.v4.new_code_cell("""# Interactive scatter plot
if 'rating' in df.columns and 'price' in df.columns and 'genre' in df.columns:
    fig = px.scatter(
        df.head(1000),  # Sample for performance
        x='price',
        y='rating',
        color='genre',
        size='price',
        hover_data=['book_name', 'author'],
        title='Interactive: Price vs Rating by Genre (Sample 1000 books)',
        labels={'price': 'Price ($)', 'rating': 'Rating'}
    )
    fig.show()"""),
        
        nbf.v4.new_code_cell("""# Interactive bar chart - Top genres
if 'genre' in df.columns:
    genre_counts = df['genre'].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation='h',
            marker=dict(
                color=genre_counts.values,
                colorscale='Viridis'
            )
        )
    ])
    
    fig.update_layout(
        title='Top 10 Genres (Interactive)',
        xaxis_title='Number of Books',
        yaxis_title='Genre',
        height=500
    )
    
    fig.show()"""),
        
        nbf.v4.new_markdown_cell("## 10. Answer Key Questions"),
        nbf.v4.new_markdown_cell("### Question 1: What are the most popular genres?"),
        nbf.v4.new_code_cell("""if 'genre' in df.columns:
    top_10_genres = df['genre'].value_counts().head(10)
    print("Top 10 Most Popular Genres:")
    for i, (genre, count) in enumerate(top_10_genres.items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{i}. {genre}: {count} books ({percentage:.1f}%)")"""),
        
        nbf.v4.new_markdown_cell("### Question 2: Which authors have the highest-rated books?"),
        nbf.v4.new_code_cell("""if 'author' in df.columns and 'rating' in df.columns:
    # Authors with avg rating >= 4.5 and at least 2 books
    author_ratings = df.groupby('author').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    author_ratings.columns = ['author', 'avg_rating', 'book_count']
    
    top_authors = author_ratings[
        (author_ratings['avg_rating'] >= 4.5) & 
        (author_ratings['book_count'] >= 2)
    ].sort_values('avg_rating', ascending=False).head(10)
    
    print("Top 10 Highest Rated Authors (avg ≥ 4.5, min 2 books):")
    for i, row in enumerate(top_authors.itertuples(), 1):
        print(f"{i}. {row.author}: {row.avg_rating:.2f} stars ({row.book_count} books)")"""),
        
        nbf.v4.new_markdown_cell("### Question 3: What is the average rating distribution?"),
        nbf.v4.new_code_cell("""if 'rating' in df.columns:
    rating_ranges = pd.cut(df['rating'], bins=[0, 2, 3, 4, 5], labels=['Poor (0-2)', 'Fair (2-3)', 'Good (3-4)', 'Excellent (4-5)'])
    rating_dist = rating_ranges.value_counts().sort_index()
    
    print("Rating Distribution:")
    for category, count in rating_dist.items():
        percentage = (count / len(df)) * 100
        print(f"{category}: {count} books ({percentage:.1f}%)")
    
    # Pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(rating_dist.values, labels=rating_dist.index, autopct='%1.1f%%', startangle=90)
    plt.title('Rating Distribution by Category')
    plt.axis('equal')
    plt.show()"""),
        
        nbf.v4.new_markdown_cell("### Question 4: Price vs Rating relationship"),
        nbf.v4.new_code_cell("""if 'price' in df.columns and 'rating' in df.columns:
    # Create price bins
    df['price_range'] = pd.cut(df['price'], 
                                bins=[0, 10, 20, 30, 100], 
                                labels=['$0-10', '$10-20', '$20-30', '$30+'])
    
    price_rating = df.groupby('price_range')['rating'].agg(['mean', 'count'])
    
    print("Average Rating by Price Range:")
    print(price_rating)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(price_rating)), price_rating['mean'], alpha=0.7)
    plt.xticks(range(len(price_rating)), price_rating.index)
    plt.xlabel('Price Range')
    plt.ylabel('Average Rating')
    plt.title('Average Rating by Price Range')
    plt.grid(axis='y', alpha=0.3)
    
    for i, row in enumerate(price_rating.itertuples()):
        plt.text(i, row.mean + 0.05, f"n={row.count}", ha='center')
    
    plt.tight_layout()
    plt.show()"""),
        
        nbf.v4.new_markdown_cell("## 11. Save Visualizations"),
        nbf.v4.new_code_cell("""from pathlib import Path

# Create output directory
output_dir = Path('../outputs/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print("✅ All visualizations can be saved manually or programmatically")
print(f"Output directory created: {output_dir}")"""),
        
        nbf.v4.new_markdown_cell("""## Summary & Key Insights

### Key Findings:

1. **Genre Insights:**
   - [Top 3 genres]
   - [Genre distribution pattern]

2. **Rating Insights:**
   - Average rating: [X.XX]
   - Most books rated between [X-X]
   - [Rating trends]

3. **Author Insights:**
   - [Top authors findings]
   - [Author rating patterns]

4. **Price Insights:**
   - Average price: $[XX.XX]
   - [Price-rating relationship]

5. **Review Insights:**
   - [Review count patterns]
   - [Correlation with ratings]

### Business Recommendations:

1. Focus on popular genres: [List]
2. Promote highly-rated authors: [List]
3. Optimize pricing strategy based on: [Findings]
4. Target book categories: [Recommendations]

---

**Notebook 03 Complete** ✅"""),
    ]
    
    return nb


def save_notebook(nb, filename, output_dir='../notebooks'):
    """Save notebook to file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    logger.info(f"✅ Created: {filepath}")


def main():
    """Create all notebooks"""
    
    print("="*60)
    print("CREATING JUPYTER NOTEBOOKS")
    print("="*60)
    
    notebooks = [
        (create_notebook_01_data_exploration(), '01_data_exploration.ipynb'),
        (create_notebook_02_data_cleaning(), '02_data_cleaning.ipynb'),
        (create_notebook_03_eda_analysis(), '03_eda_analysis.ipynb'),
    ]
    
    for nb, filename in notebooks:
        save_notebook(nb, filename)
    
    print("\n" + "="*60)
    print("✅ NOTEBOOKS CREATED SUCCESSFULLY!")
    print("="*60)
    print("\nCreated notebooks:")
    for _, filename in notebooks:
        print(f"  - notebooks/{filename}")
    
    print("\nNext: Create remaining notebooks (04-07)")


if __name__ == "__main__":
    main()
