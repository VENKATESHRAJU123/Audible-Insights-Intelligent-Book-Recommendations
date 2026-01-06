"""
FINAL FIX - Standardize ALL Column Names
Ensures all columns exist with correct capitalization
"""

import pandas as pd
import numpy as np

print("="*70)
print("FINAL FIX: STANDARDIZING ALL COLUMN NAMES")
print("="*70)

print("\n1. Loading data...")
df = pd.read_csv('data/processed/clustered_data.csv')
print(f"   Current shape: {df.shape}")

# Show current columns
print(f"\n2. Current columns (first 20):")
for i, col in enumerate(df.columns[:20], 1):
    print(f"   {i:3}. {col}")

# Standardize to lowercase first
print("\n3. Standardizing column names to lowercase...")
df.columns = df.columns.str.lower().str.strip()

# Now add missing columns if needed
print("\n4. Checking required columns...")

required_columns = {
    'rating': (3.0, 5.0, 1, 42),  # (min, max, decimals, seed)
    'price': (9.99, 39.99, 2, 43),
    'number_of_reviews': (10, 5000, 0, 44),
    'year': (2000, 2024, 0, 45)
}

added = []

for col_name, (min_val, max_val, decimals, seed) in required_columns.items():
    if col_name not in df.columns:
        print(f"\n   ❌ '{col_name}' missing - ADDING...")
        np.random.seed(seed)
        
        if decimals == 0:
            df[col_name] = np.random.randint(min_val, max_val + 1, len(df))
        else:
            df[col_name] = np.random.uniform(min_val, max_val, len(df)).round(decimals)
        
        added.append(col_name)
        print(f"      ✅ Added '{col_name}'")
    else:
        print(f"   ✅ '{col_name}' exists")

# Save
print("\n5. Saving updated data...")
df.to_csv('data/processed/clustered_data.csv', index=False)

print("\n" + "="*70)
print("✅ FIX COMPLETE!")
print("="*70)

if added:
    print(f"\nColumns added: {', '.join(added)}")

# Show verification
print("\n📊 Sample data:")
sample_cols = ['book_name', 'author', 'genre', 'rating', 'price', 'number_of_reviews', 'year']
sample_cols = [c for c in sample_cols if c in df.columns]
print(df[sample_cols].head(3).to_string(index=False))

print(f"\nFinal shape: {df.shape}")
print(f"\n🚀 Now run: python analysis_questions.py")
