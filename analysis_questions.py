"""
Analysis Questions - Simplified with Lowercase Columns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

output_dir = Path("outputs/analysis")
output_dir.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load data with lowercase columns"""
    df = pd.read_csv('data/processed/clustered_data.csv')
    df.columns = df.columns.str.lower().str.strip()
    print(f"✅ Data loaded: {df.shape}")
    return df


def question1_popular_genres(df):
    """Question 1: Most popular genres"""
    print("\n" + "="*70)
    print("QUESTION 1: MOST POPULAR GENRES")
    print("="*70)
    
    if 'genre' not in df.columns:
        print("❌ Genre column not found")
        return
    
    genre_counts = df['genre'].value_counts().head(15)
    
    print(f"\n📊 Top 15 Most Popular Genres:\n")
    for i, (genre, count) in enumerate(genre_counts.items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{i:2}. {genre:30} {count:5} ({percentage:5.1f}%)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = sns.color_palette("viridis", len(genre_counts))
    ax1.barh(range(len(genre_counts)), genre_counts.values, color=colors)
    ax1.set_yticks(range(len(genre_counts)))
    ax1.set_yticklabels(genre_counts.index)
    ax1.invert_yaxis()
    ax1.set_xlabel('Books')
    ax1.set_title('Top 15 Genres', fontweight='bold')
    
    ax2.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
    ax2.set_title('Genre Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'q1_popular_genres.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir / 'q1_popular_genres.png'}")
    plt.close()


def question2_highest_rated_authors(df):
    """Question 2: Highest rated authors"""
    print("\n" + "="*70)
    print("QUESTION 2: HIGHEST-RATED AUTHORS")
    print("="*70)
    
    if 'author' not in df.columns or 'rating' not in df.columns:
        print("❌ Required columns not found")
        return
    
    author_stats = df.groupby('author').agg({
        'book_name': 'count',
        'rating': ['mean', 'std']
    }).round(2)
    
    author_stats.columns = ['book_count', 'avg_rating', 'rating_std']
    author_stats = author_stats.reset_index()
    
    top_authors = author_stats[author_stats['book_count'] >= 3].sort_values('avg_rating', ascending=False).head(20)
    
    print(f"\n📊 Top 20 Highest Rated Authors (min 3 books):\n")
    for i, row in enumerate(top_authors.itertuples(), 1):
        print(f"{i:2}. {row.author:30} {row.avg_rating:.2f}⭐ ({row.book_count} books)")
    
    print("\n✅ Analysis complete")


def question3_rating_distribution(df):
    """Question 3: Rating distribution"""
    print("\n" + "="*70)
    print("QUESTION 3: RATING DISTRIBUTION")
    print("="*70)
    
    if 'rating' not in df.columns:
        print("❌ Rating column not found")
        return
    
    stats = df['rating'].describe()
    
    print(f"\n📊 Rating Statistics:\n")
    print(f"Mean:     {stats['mean']:.2f}⭐")
    print(f"Median:   {stats['50%']:.2f}⭐")
    print(f"Std Dev:  {stats['std']:.2f}")
    print(f"Min:      {stats['min']:.2f}⭐")
    print(f"Max:      {stats['max']:.2f}⭐")
    
    print("\n✅ Analysis complete")


def question4_publication_trends(df):
    """Question 4: Publication trends"""
    print("\n" + "="*70)
    print("QUESTION 4: PUBLICATION YEAR TRENDS")
    print("="*70)
    
    if 'year' not in df.columns:
        print("⚠️  Year column not found - skipping")
        return
    
    year_stats = df.groupby('year').size().sort_index()
    
    print(f"\n📊 Books by Year:\n")
    for year, count in year_stats.items():
        print(f"{year}: {count} books")
    
    print(f"\n✅ Analysis complete")


def question5_ratings_vs_reviews(df):
    """Question 5: Ratings vs reviews"""
    print("\n" + "="*70)
    print("QUESTION 5: RATINGS vs REVIEW COUNTS")
    print("="*70)
    
    if 'rating' not in df.columns:
        print(f"❌ Rating column not found")
        print(f"Available columns: {[c for c in df.columns if 'rat' in c]}")
        return
    
    if 'number_of_reviews' not in df.columns:
        print(f"❌ Reviews column not found")
        print(f"Available columns: {[c for c in df.columns if 'review' in c]}")
        return
    
    corr = df[['rating', 'number_of_reviews']].corr().iloc[0, 1]
    print(f"\n📊 Correlation: {corr:.4f}")
    
    # Create categories
    df['review_category'] = pd.cut(df['number_of_reviews'], 
                                   bins=[0, 50, 200, 500, 1000, 10000],
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    stats = df.groupby('review_category').agg({
        'rating': ['mean', 'count'],
        'number_of_reviews': 'mean'
    }).round(2)
    
    print(f"\n📊 Statistics by Review Count:\n")
    print(stats)
    
    print("\n✅ Analysis complete")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("BOOK DATASET ANALYSIS")
    print("="*70)
    
    df = load_data()
    
    # Print available columns for debugging
    print(f"\nKey columns available:")
    for col in ['book_name', 'author', 'genre', 'rating', 'price', 'number_of_reviews', 'year']:
        status = "✅" if col in df.columns else "❌"
        print(f"  {status} {col}")
    
    question1_popular_genres(df)
    question2_highest_rated_authors(df)
    question3_rating_distribution(df)
    question4_publication_trends(df)
    question5_ratings_vs_reviews(df)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nVisualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
