"""
Medium-Level Analysis Questions (Combination of Tables/Models)

1. Which books are frequently clustered together based on descriptions?
2. How does genre similarity affect book recommendations?
3. What is the effect of author popularity on book ratings?
4. Which combination of features provides the most accurate recommendations?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# For Question 4
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


# ---------- COMMON HELPERS ----------

def load_data(path="data/processed/clustered_data.csv") -> pd.DataFrame:
    """Load clustered data and standardize column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    return df


def get_col(df: pd.DataFrame, candidates):
    """Return first column from `candidates` that exists in df (all lowercase)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================================================
# Q1. Which books are frequently clustered together
#     based on descriptions?
# =========================================================

def q1_books_in_same_cluster(df: pd.DataFrame, min_cluster_size=5, top_n=5):
    """
    STEP BY STEP:
    1. Find the cluster column (k-means labels).
    2. Count how many books in each cluster.
    3. Filter clusters with at least `min_cluster_size` books.
    4. For each large cluster, list top N books (e.g. by rating).
    """

    cluster_col = get_col(df, ["cluster_kmeans", "cluster"])
    book_col    = get_col(df, ["book_name"])
    rating_col  = get_col(df, ["rating"])

    if cluster_col is None or book_col is None:
        raise ValueError("Need 'cluster_kmeans' (or 'cluster') and 'book_name' columns.")

    # 1–2. cluster sizes
    cluster_sizes = df[cluster_col].value_counts().sort_index()
    print("Cluster sizes:\n", cluster_sizes, "\n")

    # 3. filter “large” clusters
    large_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
    print(f"Clusters with ≥ {min_cluster_size} books:", large_clusters, "\n")

    results = []

    # 4. list books inside each large cluster
    for cid in large_clusters:
        cluster_df = df[df[cluster_col] == cid].copy()
        if rating_col:
            cluster_df = cluster_df.sort_values(rating_col, ascending=False)

        # top N books in this cluster
        top_books = cluster_df.head(top_n)[[book_col]]
        if rating_col in cluster_df.columns:
            top_books[rating_col] = cluster_df.head(top_n)[rating_col].values

        print(f"Cluster {cid} – top {len(top_books)} books:")
        print(top_books.to_string(index=False), "\n")

        results.append((cid, cluster_df))

    return results  # you can inspect or use further


# =========================================================
# Q2. How does genre similarity affect book recommendations?
# =========================================================

def q2_genre_similarity_vs_clusters(df: pd.DataFrame):
    """
    STEP BY STEP:
    1. For each cluster, compute genre distribution.
    2. Dominant genre share = (#books in most common genre / cluster size).
    3. High dominant share ⇒ recommendations from same cluster are often same-genre.
    """

    cluster_col = get_col(df, ["cluster_kmeans", "cluster"])
    genre_col   = get_col(df, ["genre"])

    if cluster_col is None or genre_col is None:
        raise ValueError("Need 'cluster_kmeans' (or 'cluster') and 'genre' columns.")

    stats = []

    for cid, g in df.groupby(cluster_col):
        total = len(g)
        genre_counts = g[genre_col].value_counts()

        dominant_genre = genre_counts.index[0]
        dominant_count = genre_counts.iloc[0]
        dominant_share = dominant_count / total  # 0–1

        stats.append({
            "cluster_id": cid,
            "cluster_size": total,
            "dominant_genre": dominant_genre,
            "dominant_share": round(dominant_share * 100, 2)
        })

    stats_df = pd.DataFrame(stats).sort_values("dominant_share", ascending=False)
    print("Genre homogeneity by cluster:")
    print(stats_df.to_string(index=False), "\n")

    print("Average dominant-genre share: "
          f"{stats_df['dominant_share'].mean():.2f}%")
    print("Clusters with ≥80% same genre: ",
          (stats_df['dominant_share'] >= 80).sum())

    return stats_df


# =========================================================
# Q3. Effect of author popularity on book ratings
# =========================================================

def q3_author_popularity_vs_rating(df: pd.DataFrame):
    """
    STEP BY STEP:
    1. Define author popularity:
       - number of books
       - (optionally) total review count.
    2. Compute average rating per author.
    3. Correlate popularity with average rating.
    """

    author_col  = get_col(df, ["author"])
    rating_col  = get_col(df, ["rating"])
    reviews_col = get_col(df, ["number_of_reviews"])

    if author_col is None or rating_col is None:
        raise ValueError("Need 'author' and 'rating' columns.")

    agg = {
        rating_col: ["mean", "count"]
    }
    if reviews_col:
        agg[reviews_col] = ["sum"]

    author_stats = df.groupby(author_col).agg(agg).round(2)
    author_stats.columns = ["avg_rating", "book_count"] + (["total_reviews"] if reviews_col else [])
    author_stats = author_stats.reset_index()

    print("Author statistics (first 10):")
    print(author_stats.head(10).to_string(index=False), "\n")

    # Correlations
    correlations = {}
    correlations["books_vs_rating"] = author_stats[["book_count", "avg_rating"]].corr().iloc[0, 1]
    if reviews_col:
        correlations["reviews_vs_rating"] = author_stats[["total_reviews", "avg_rating"]].corr().iloc[0, 1]

    print("Correlations:")
    for name, val in correlations.items():
        print(f"  {name}: {val:.4f}")
    print()

    return author_stats, correlations


# =========================================================
# Q4. Which combination of features gives most accurate
#     rating predictions? (proxy for recommendation quality)
# =========================================================

def q4_feature_importance(df: pd.DataFrame):
    """
    STEP BY STEP:
    1. Select features (numeric + categorical).
    2. Build preprocessing pipeline (OneHot for categorical).
    3. Train RandomForestRegressor to predict rating.
    4. Evaluate R² as a rough accuracy metric.
    5. Examine feature importances (which features matter most).
    """

    rating_col = get_col(df, ["rating"])
    genre_col  = get_col(df, ["genre"])
    author_col = get_col(df, ["author"])
    price_col  = get_col(df, ["price"])
    reviews_col = get_col(df, ["number_of_reviews"])
    year_col   = get_col(df, ["year"])  # may or may not exist

    if rating_col is None:
        raise ValueError("Need 'rating' column.")

    numeric_features = [c for c in [price_col, reviews_col, year_col] if c is not None]
    categorical_features = [c for c in [genre_col, author_col] if c is not None]

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features, "\n")

    feature_cols = numeric_features + categorical_features
    if not feature_cols:
        raise ValueError("No feature columns found.")

    # Drop rows with missing rating
    model_df = df.dropna(subset=[rating_col]).copy()
    X = model_df[feature_cols]
    y = model_df[rating_col]

    # Build preprocessing
    transformers = []
    if numeric_features:
        transformers.append(("num", "passthrough", numeric_features))
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("rf", rf)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"RandomForest R² (rating prediction): {r2:.3f}\n")

    # --- Feature importances aggregated by original column ---

    importances = rf.feature_importances_
    pre = pipe.named_steps["pre"]

    rows = []

    # If we have numeric and categorical
    if numeric_features and categorical_features:
        num_len = len(numeric_features)
        # numeric part
        for col, imp in zip(numeric_features, importances[:num_len]):
            rows.append({"feature": col, "importance": imp})

        # categorical part aggregated
        ohe = pre.named_transformers_["cat"]
        cat_feature_names = ohe.get_feature_names_out(categorical_features)

        cat_imp_map = {col: 0.0 for col in categorical_features}
        for cat_name, imp in zip(cat_feature_names, importances[num_len:]):
            orig_col = cat_name.split("_")[0]  # e.g. 'genre_xxx' -> 'genre'
            cat_imp_map[orig_col] += imp
        for col, imp in cat_imp_map.items():
            rows.append({"feature": col, "importance": imp})

    # Only numeric OR only categorical
    else:
        if numeric_features:
            for col, imp in zip(numeric_features, importances):
                rows.append({"feature": col, "importance": imp})
        else:  # only categorical
            ohe = pre.named_transformers_["cat"]
            cat_feature_names = ohe.get_feature_names_out(categorical_features)
            cat_imp_map = {col: 0.0 for col in categorical_features}
            for cat_name, imp in zip(cat_feature_names, importances):
                orig_col = cat_name.split("_")[0]
                cat_imp_map[orig_col] += imp
            for col, imp in cat_imp_map.items():
                rows.append({"feature": col, "importance": imp})

    imp_df = pd.DataFrame(rows)
    imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()
    imp_df = imp_df.sort_values("importance_norm", ascending=False)

    print("Feature importance (aggregated by original column):")
    print(imp_df.to_string(index=False), "\n")

    return r2, imp_df


# =========================================================
# MAIN: run all questions
# =========================================================

def main():
    df = load_data()

    print("\n" + "="*70)
    print("Q1: BOOKS CLUSTERED TOGETHER")
    print("="*70)
    q1_books_in_same_cluster(df, min_cluster_size=5, top_n=5)

    print("\n" + "="*70)
    print("Q2: GENRE SIMILARITY & CLUSTERS")
    print("="*70)
    q2_genre_similarity_vs_clusters(df)

    print("\n" + "="*70)
    print("Q3: AUTHOR POPULARITY vs RATINGS")
    print("="*70)
    q3_author_popularity_vs_rating(df)

    print("\n" + "="*70)
    print("Q4: FEATURE COMBINATION IMPORTANCE")
    print("="*70)
    q4_feature_importance(df)


if __name__ == "__main__":
    main()
