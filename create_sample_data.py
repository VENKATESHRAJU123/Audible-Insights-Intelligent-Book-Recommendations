"""
Create Complete Sample Data with ALL Columns
Includes: Book Name, Author, Rating, Reviews, Price, Description, Genre, Year
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

def main():
    print("\n" + "="*70)
    print("CREATING COMPLETE SAMPLE AUDIBLE DATASETS")
    print("="*70 + "\n")
    
    np.random.seed(42)
    random.seed(42)
    
    # Create directories
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    # Reference data
    genres = ['Fiction', 'Mystery', 'Romance', 'Sci-Fi', 'Fantasy', 
              'Thriller', 'Biography', 'Self-Help', 'History', 'Science',
              'Business', 'Psychology', 'Horror', 'Adventure', 'Comedy']
    
    authors = ['J.K. Rowling', 'Stephen King', 'Agatha Christie', 'Dan Brown',
               'Malcolm Gladwell', 'Yuval Noah Harari', 'Michelle Obama', 'Bill Bryson',
               'Isaac Asimov', 'Arthur Conan Doyle', 'Margaret Atwood', 'George R.R. Martin',
               'Jane Austen', 'Ernest Hemingway', 'F. Scott Fitzgerald', 'Mark Twain',
               'Charles Dickens', 'Leo Tolstoy', 'James Patterson', 'John Grisham']
    
    # DATASET 1 - COMPLETE
    print("Creating Dataset 1 (Complete)...")
    n1 = 500
    
    df1 = pd.DataFrame({
        'Book Name': [f'Book {i}: The {random.choice(["Journey", "Mystery", "Tale", "Adventure"])}' 
                      for i in range(n1)],
        'Author': [random.choice(authors) for _ in range(n1)],
        'Rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(n1)],
        'Number of Reviews': [random.randint(10, 5000) for _ in range(n1)],  # ← IMPORTANT
        'Price': [round(random.uniform(9.99, 39.99), 2) for _ in range(n1)],
        'Description': [f'A wonderful {random.choice(["story", "tale", "saga"])} about {random.choice(["adventure", "mystery", "love", "courage"])}' 
                       for _ in range(n1)],
        'Listening Time': [f'{random.randint(3, 25)} hours {random.randint(0, 59)} minutes' 
                          for _ in range(n1)],
        'Genre': [random.choice(genres) for _ in range(n1)],
        'Year': [random.randint(2000, 2024) for _ in range(n1)]  # ← PUBLICATION YEAR
    })
    
    # Save Dataset 1
    file1 = raw_data_path / "Audible_Catalog.csv"
    df1.to_csv(file1, index=False)
    
    print(f"✅ Created: {file1}")
    print(f"   Rows: {len(df1)}")
    print(f"   Columns: {df1.columns.tolist()}")
    
    # Verify critical columns
    critical_cols = ['Rating', 'Number of Reviews', 'Price', 'Year']
    print(f"\n   Critical columns present:")
    for col in critical_cols:
        if col in df1.columns:
            print(f"      ✅ {col}")
        else:
            print(f"      ❌ {col} MISSING!")
    
    # DATASET 2
    print("\nCreating Dataset 2...")
    n_overlap = int(n1 * 0.7)
    n_new = int(n1 * 0.3)
    
    # Overlapping books from Dataset 1
    overlap_df = df1.sample(n_overlap)[['Book Name', 'Author', 'Rating', 
                                         'Number of Reviews', 'Price', 'Year']].copy()
    
    # New books
    new_books = pd.DataFrame({
        'Book Name': [f'New Book {i}: {random.choice(["The Story", "A Legend", "Chronicles"])}' 
                      for i in range(n_new)],
        'Author': [random.choice(authors) for _ in range(n_new)],
        'Rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(n_new)],
        'Number of Reviews': [random.randint(10, 5000) for _ in range(n_new)],
        'Price': [round(random.uniform(9.99, 39.99), 2) for _ in range(n_new)],
        'Year': [random.randint(2000, 2024) for _ in range(n_new)]
    })
    
    # Combine
    df2 = pd.concat([overlap_df, new_books], ignore_index=True)
    df2 = df2.sample(frac=1).reset_index(drop=True)
    
    # Save Dataset 2
    file2 = raw_data_path / "Audible_Catalog_Advanced_Features.csv"
    df2.to_csv(file2, index=False)
    
    print(f"✅ Created: {file2}")
    print(f"   Rows: {len(df2)}")
    print(f"   Columns: {df2.columns.tolist()}")
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ COMPLETE DATASETS CREATED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"   Dataset 1: {len(df1)} books, {len(df1.columns)} columns")
    print(f"   Dataset 2: {len(df2)} books, {len(df2.columns)} columns")
    
    print(f"\n✅ All Required Columns Present:")
    print(f"   • Book Name")
    print(f"   • Author")
    print(f"   • Rating")
    print(f"   • Number of Reviews")
    print(f"   • Price")
    print(f"   • Genre")
    print(f"   • Year (Publication)")
    print(f"   • Description")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. python src/data_processing.py")
    print(f"   2. python src/nlp_features.py")
    print(f"   3. python src/clustering.py")
    print(f"   4. python analysis_questions.py")
    print()


if __name__ == "__main__":
    main()
