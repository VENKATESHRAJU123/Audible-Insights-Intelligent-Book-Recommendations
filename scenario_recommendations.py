"""
Scenario-Based Recommendations

1. A new user likes science fiction books. Which top 5 books should be recommended?
2. For a user who has previously rated thrillers highly, recommend similar books.
3. Identify books that are highly rated but have low popularity (hidden gems).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# 0. Common helpers
# -------------------------------------------------------------------

def load_data(path="data/processed/clustered_data.csv") -> pd.DataFrame:
    """
    STEP 0: Load the clustered dataset and normalize column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    return df


def get_col(df: pd.DataFrame, candidates):
    """
    Return the first candidate column that exists in df (all lowercase).
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -------------------------------------------------------------------
# 1. Scenario 1 – New user likes science fiction (Sci‑Fi)
# -------------------------------------------------------------------

def scenario1_scifi_top5(df: pd.DataFrame):
    """
    STEP-BY-STEP:
      1. Find `genre`, `rating`, and `book_name` columns.
      2. Filter books whose genre contains 'sci-fi' / 'science fiction'.
      3. Sort these books by rating (and optionally reviews).
      4. Return top 5 as recommendations.
    """

    genre_col   = get_col(df, ["genre"])
    rating_col  = get_col(df, ["rating"])
    reviews_col = get_col(df, ["number_of_reviews"])
    book_col    = get_col(df, ["book_name"])
    author_col  = get_col(df, ["author"])

    if not all([genre_col, rating_col, book_col]):
        raise ValueError("Need 'genre', 'rating', and 'book_name' columns for Scenario 1.")

    # 1. Filter Sci‑Fi / Science Fiction books (case-insensitive)
    mask_scifi = df[genre_col].str.contains("sci-fi|science fiction|science-fiction",
                                            case=True, na=False, regex=True)
    scifi_df = df[mask_scifi].copy()

    if scifi_df.empty:
        print("No science fiction books found in the dataset.")
        return

    # 2. Sort by rating (desc), and by number_of_reviews if available
    sort_cols = [rating_col]
    sort_asc  = [False]
    if reviews_col:
        sort_cols.append(reviews_col)
        sort_asc.append(False)

    scifi_sorted = scifi_df.sort_values(sort_cols, ascending=sort_asc)

    # 3. Take top 5
    top5 = scifi_sorted.head(5)

    display_cols = [book_col]
    for c in [author_col, genre_col, rating_col, reviews_col]:
        if c in df.columns:
            display_cols.append(c)

    print("\n" + "="*70)
    print("SCENARIO 1: New user likes Science Fiction – Top 5 Recommendations")
    print("="*70)
    print(top5[display_cols].to_string(index=False))
    print()

    return top5[display_cols]


# -------------------------------------------------------------------
# 2. Scenario 2 – User rated thrillers highly → similar books
# -------------------------------------------------------------------

def scenario2_thriller_similar(df: pd.DataFrame):
    """
    STEP-BY-STEP:
      1. Identify thriller books:
         - genre contains 'thriller'.
      2. Among thrillers, pick those with high ratings (e.g. ≥ 4.0) –
         this approximates "user has rated thrillers highly".
      3. Use clusters to find similar books:
         - Take clusters that contain high-rated thrillers.
         - Within those clusters, recommend other books (not necessarily only thrillers).
    """

    genre_col    = get_col(df, ["genre"])
    rating_col   = get_col(df, ["rating"])
    book_col     = get_col(df, ["book_name"])
    author_col   = get_col(df, ["author"])
    cluster_col  = get_col(df, ["cluster_kmeans", "cluster"])
    reviews_col  = get_col(df, ["number_of_reviews"])

    if not all([genre_col, rating_col, book_col, cluster_col]):
        raise ValueError("Need 'genre', 'rating', 'book_name', and 'cluster_kmeans'/'cluster' columns.")

    # 1. Thriller books
    mask_thriller = df[genre_col].str.contains("thriller", case=False, na=False)
    thriller_df = df[mask_thriller].copy()

    if thriller_df.empty:
        print("No thriller books found in the dataset.")
        return

    # 2. High-rated thrillers (e.g. rating ≥ 4.0)
    high_thriller_df = thriller_df[thriller_df[rating_col] >= 4.0]

    if high_thriller_df.empty:
        print("No high-rated thrillers (rating >= 4.0) found.")
        return

    # Clusters that contain at least one high-rated thriller
    thriller_clusters = high_thriller_df[cluster_col].unique().tolist()

    # 3. For each such cluster, recommend other high-rated books
    recs_list = []

    for cid in thriller_clusters:
        cluster_books = df[df[cluster_col] == cid].copy()
        # Exclude the original thrillers (if we only want "new" books)
        cluster_books = cluster_books[~cluster_books.index.isin(high_thriller_df.index)]

        # Filter by rating threshold for recommendations, e.g. ≥ 4.0
        cluster_recs = cluster_books[cluster_books[rating_col] >= 4.0]
        if reviews_col in cluster_recs.columns:
            cluster_recs = cluster_recs.sort_values(
                [rating_col, reviews_col], ascending=[False, False]
            )
        else:
            cluster_recs = cluster_recs.sort_values(rating_col, ascending=False)

        recs_list.append(cluster_recs)

    # Combine and deduplicate
    if recs_list:
        all_recs = pd.concat(recs_list).drop_duplicates(subset=[book_col])
    else:
        print("No similar books found in thriller clusters.")
        return

    top_n = 10  # you can adjust
    top_recs = all_recs.head(top_n)

    display_cols = [book_col]
    for c in [author_col, genre_col, rating_col, reviews_col]:
        if c in df.columns:
            display_cols.append(c)

    print("\n" + "="*70)
    print("SCENARIO 2: User likes Thrillers & has rated them highly – Similar Recommendations")
    print("="*70)
    print(top_recs[display_cols].to_string(index=False))
    print()

    return top_recs[display_cols]


# -------------------------------------------------------------------
# 3. Scenario 3 – Hidden gems (high rating, low popularity)
# -------------------------------------------------------------------

def scenario3_hidden_gems(df: pd.DataFrame,
                          rating_threshold=4.5,
                          max_review_count=50):
    """
    STEP-BY-STEP:
      1. Define "highly rated" → rating ≥ rating_threshold.
      2. Define "low popularity" → number_of_reviews ≤ max_review_count.
      3. Filter books that satisfy both conditions.
      4. Sort by rating (desc) and reviews (asc), return top N.
    """

    rating_col  = get_col(df, ["rating"])
    reviews_col = get_col(df, ["number_of_reviews"])
    book_col    = get_col(df, ["book_name"])
    author_col  = get_col(df, ["author"])
    genre_col   = get_col(df, ["genre"])

    if not all([rating_col, reviews_col, book_col]):
        raise ValueError("Need 'rating', 'number_of_reviews', and 'book_name' columns.")

    # 1 + 2. Filter high rating & low reviews
    mask_high   = df[rating_col] >= rating_threshold
    mask_lowpop = df[reviews_col] <= max_review_count

    hidden_df = df[mask_high & mask_lowpop].copy()

    if hidden_df.empty:
        print(f"No 'hidden gems' found with rating ≥ {rating_threshold} and reviews ≤ {max_review_count}.")
        return

    # 3. Sort hidden gems: high rating first, then fewer reviews
    hidden_sorted = hidden_df.sort_values(
        [rating_col, reviews_col],
        ascending=[False, True]
    )

    top_n = 20
    top_hidden = hidden_sorted.head(top_n)

    display_cols = [book_col]
    for c in [author_col, genre_col, rating_col, reviews_col]:
        if c in df.columns:
            display_cols.append(c)

    print("\n" + "="*70)
    print(f"SCENARIO 3: Hidden Gems – Rating ≥ {rating_threshold}, Reviews ≤ {max_review_count}")
    print("="*70)
    print(top_hidden[display_cols].to_string(index=False))
    print()

    return top_hidden[display_cols]


# -------------------------------------------------------------------
# MAIN – Run all 3 scenarios
# -------------------------------------------------------------------

def main():
    # Step 0: Load data
    df = load_data()

    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", df.columns.tolist(), "\n")

    # Scenario 1
    scenario1_scifi_top5(df)

    # Scenario 2
    scenario2_thriller_similar(df)

    # Scenario 3
    scenario3_hidden_gems(df, rating_threshold=4.5, max_review_count=50)


if __name__ == "__main__":
    main()
