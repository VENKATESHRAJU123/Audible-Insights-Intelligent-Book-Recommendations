"""
Quick Data Creator - Run this first!
Creates sample Audible datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

print("Creating sample Audible datasets...")

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Sample data
genres = ['Fiction', 'Mystery', 'Romance', 'Sci-Fi', 'Fantasy', 'Thriller', 
          'Biography', 'Self-Help', 'History', 'Science']

authors = [
    'J.K. Rowling', 'Stephen King', 'Agatha Christie', 'Dan Brown',
    'Malcolm Gladwell', 'Yuval Noah Harari', 'Michelle Obama', 'Bill Bryson',
    'Isaac Asimov', 'Arthur Conan Doyle', 'Margaret Atwood', 'George R.R. Martin',
    'Jane Austen', 'Ernest Hemingway', 'F. Scott Fitzgerald', 'Mark Twain'
]

# CREATE DATASET 1
print("Creating Dataset 1...")
n1 = 500

df1 = pd.DataFrame({
    'Book Name': [f'Book {i}: {random.choice(["The", "A", "An"])} {random.choice(["Journey", "Mystery", "Tale", "Adventure"])}' for i in range(n1)],
    'Author': [random.choice(authors) for _ in range(n1)],
    'Rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(n1)],
    'Number of Reviews': [random.randint(10, 5000) for _ in range(n1)],
    'Price': [round(random.uniform(9.99, 39.99), 2) for _ in range(n1)],
    'Description': [f'A wonderful story about {random.choice(["adventure", "mystery", "love", "courage"])}' for _ in range(n1)],
    'Listening Time': [f'{random.randint(3, 20)} hours' for _ in range(n1)],
    'Genre': [random.choice(genres) for _ in range(n1)]
})

# Save Dataset 1
df1.to_csv('data/raw/Audible_Catalog.csv', index=False)
print(f"✅ Created: data/raw/Audible_Catalog.csv ({len(df1)} books)")

# CREATE DATASET 2
print("Creating Dataset 2...")
# Use 70% of books from dataset 1 + 30% new books
n_overlap = int(n1 * 0.7)
n_new = int(n1 * 0.3)

# Overlapping books
overlap_books = df1.sample(n_overlap)[['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Price']].copy()

# New books
new_books = pd.DataFrame({
    'Book Name': [f'New Book {i}: {random.choice(["The", "A"])} {random.choice(["Story", "Legend", "Chronicle"])}' for i in range(n_new)],
    'Author': [random.choice(authors) for _ in range(n_new)],
    'Rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(n_new)],
    'Number of Reviews': [random.randint(10, 5000) for _ in range(n_new)],
    'Price': [round(random.uniform(9.99, 39.99), 2) for _ in range(n_new)]
})

# Combine
df2 = pd.concat([overlap_books, new_books], ignore_index=True)
df2 = df2.sample(frac=1).reset_index(drop=True)  # Shuffle

# Save Dataset 2
df2.to_csv('data/raw/Audible_Catalog_Advanced_Features.csv', index=False)
print(f"✅ Created: data/raw/Audible_Catalog_Advanced_Features.csv ({len(df2)} books)")

print("\n" + "="*60)
print("✅ SUCCESS! Sample datasets created!")
print("="*60)
print("\nDataset 1 columns:", list(df1.columns))
print("Dataset 2 columns:", list(df2.columns))
print("\nNow run: python src/data_processing.py")
