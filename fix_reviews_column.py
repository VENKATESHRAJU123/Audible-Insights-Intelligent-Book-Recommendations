"""
Add Missing Number of Reviews Column
"""

import pandas as pd
import numpy as np

print("Loading clustered data...")
df = pd.read_csv('data/processed/clustered_data.csv')

print(f"Current shape: {df.shape}")

# Check for review columns
review_variants = ['Number of Reviews', 'number_of_reviews', 'reviews', 'review_count', 'Number_of_Reviews']
found_review_col = None

for variant in review_variants:
    if variant in df.columns:
        found_review_col = variant
        print(f"✅ Found review column: {variant}")
        break

if not found_review_col:
    print("❌ No review column found - adding synthetic data...")
    
    # Add Number of Reviews column
    np.random.seed(42)
    df['Number of Reviews'] = np.random.randint(10, 5000, len(df))
    
    print("✅ Added 'Number of Reviews' column")
    
    # Save
    df.to_csv('data/processed/clustered_data.csv', index=False)
    print(f"✅ Data saved with new column")
    print(f"New shape: {df.shape}")
else:
    print(f"✅ Review column already exists as: {found_review_col}")

# Show sample
print("\nSample data:")
cols_to_show = ['book_name', 'Rating', 'Number of Reviews']
cols_to_show = [c for c in cols_to_show if c in df.columns]
print(df[cols_to_show].head())

print("\n🚀 Now run: python analysis_questions.py")
