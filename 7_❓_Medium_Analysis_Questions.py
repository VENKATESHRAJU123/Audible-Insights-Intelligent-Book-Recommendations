"""
7_❓_Medium_Analysis_Questions.py

Medium-Level Analysis Questions (Combination of Tables/Models)

1. Which books are frequently clustered together based on descriptions?
2. How does genre similarity affect book recommendations?
3. What is the effect of author popularity on book ratings?
4. Which combination of features provides the most accurate recommendations?
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Make sure we can import from project root if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Medium Analysis Questions",
    page_icon="❓",
    layout="wide"
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load clustered data and normalize column names."""
    df = pd.read_csv("data/processed/clustered_data.csv")
    df.columns = df.columns.str.lower().str.strip()
    return df


def get_col(df: pd.DataFrame, candidates):
    """Return first column from candidates that exists in df (lowercase)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ===================================================================
# Q1 – Which books are frequently clustered together?
# ===================================================================

def show_q1(df: pd.DataFrame):
    st.markdown("## 1️⃣ Books Clustered Together (Based on Descriptions)")

    cluster_col = get_col(df, ["cluster_kmeans", "cluster"])
    book_col    = get_col(df, ["book_name"])
    rating_col  = get_col(df, ["rating"])
    author_col  = get_col(df, ["author"])
    genre_col   = get_col(df, ["genre"])

    if cluster_col is None or book_col is None:
        st.error("Need 'cluster_kmeans' (or 'cluster') and 'book_name' columns in clustered_data.csv.")
        return

    # Cluster size overview
    cluster_sizes = df[cluster_col].value_counts().sort_index()
    fig = px.bar(
        x=cluster_sizes.index,
        y=cluster_sizes.values,
        labels={"x": "Cluster ID", "y": "Number of Books"},
        title="Books per Cluster (Description-Based Clustering)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        min_size = st.slider("Minimum cluster size", 2, int(cluster_sizes.max()), 5)
        cluster_ids = cluster_sizes[cluster_sizes >= min_size].index.tolist()
        if not cluster_ids:
            st.warning("No clusters meet this minimum size; lower the threshold.")
            return

        selected_cluster = st.selectbox("Select cluster", cluster_ids)

    cluster_df = df[df[cluster_col] == selected_cluster].copy()
    if rating_col:
        cluster_df = cluster_df.sort_values(rating_col, ascending=False)

    with col2:
        st.metric("Books in selected cluster", f"{len(cluster_df):,}")

    top_n = st.slider("Top N books to display", 5, min(50, len(cluster_df)), 15)

    display_cols = [book_col]
    for c in [author_col, genre_col, rating_col]:
        if c and c in cluster_df.columns:
            display_cols.append(c)

    st.markdown(f"### Top {top_n} books in cluster `{selected_cluster}`")
    st.dataframe(
        cluster_df[display_cols].head(top_n),
        use_container_width=True,
        hide_index=True
    )

    # Within-cluster author/genre patterns
    if author_col and genre_col:
        st.markdown("#### Most Common Author–Genre Combinations in This Cluster")
        combo = (
            cluster_df.groupby([author_col, genre_col])[book_col]
            .count()
            .reset_index()
            .rename(columns={book_col: "book_count"})
            .sort_values("book_count", ascending=False)
            .head(15)
        )
        st.dataframe(combo, use_container_width=True, hide_index=True)

    st.markdown("#### Interpretation")
    st.info(
        """
        - Books inside the **same cluster** are considered similar based on description‑derived
          features (TF‑IDF + other encodings).
        - These clusters can be used for **cluster‑based recommendations**: for a given book,
          recommend other books from its cluster.
        """
    )


# ===================================================================
# Q2 – How does genre similarity affect recommendations?
# ===================================================================

def show_q2(df: pd.DataFrame):
    st.markdown("## 2️⃣ Genre Similarity and Clusters")

    cluster_col = get_col(df, ["cluster_kmeans", "cluster"])
    genre_col   = get_col(df, ["genre"])

    if cluster_col is None or genre_col is None:
        st.error("Need 'cluster_kmeans' (or 'cluster') and 'genre' columns.")
        return

    stats = []
    for cid, g in df.groupby(cluster_col):
        total = len(g)
        if total == 0:
            continue
        counts = g[genre_col].value_counts()
        dom_genre = counts.index[0]
        dom_share = counts.iloc[0] / total
        stats.append({
            "cluster_id": cid,
            "cluster_size": total,
            "dominant_genre": dom_genre,
            "dominant_share_%": round(dom_share * 100, 2),
        })

    stats_df = pd.DataFrame(stats).sort_values("dominant_share_%", ascending=False)

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of clusters", stats_df.shape[0])
    with col2:
        st.metric("Average dominant-genre share",
                  f"{stats_df['dominant_share_%'].mean():.1f}%")
    with col3:
        st.metric("Clusters ≥80% same genre",
                  int((stats_df["dominant_share_%"] >= 80).sum()))

    # Bar plot
    fig = px.bar(
        stats_df,
        x="cluster_id",
        y="dominant_share_%",
        hover_data=["cluster_size", "dominant_genre"],
        title="Genre Homogeneity per Cluster",
        labels={"cluster_id": "Cluster ID", "dominant_share_%": "Dominant Genre Share (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show cluster–genre statistics table"):
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("#### Interpretation")
    st.info(
        """
        - Clusters with a **high dominant‑genre share** (e.g. >80%) mean that
          the clustering (and hence cluster‑based recommendations) are heavily influenced by genre.
        - Mixed‑genre clusters indicate that other features (text description, ratings, etc.)
          play a larger role beyond genre.
        """
    )


# ===================================================================
# Q3 – Effect of author popularity on ratings
# ===================================================================

def show_q3(df: pd.DataFrame):
    st.markdown("## 3️⃣ Author Popularity vs Book Ratings")

    author_col  = get_col(df, ["author"])
    rating_col  = get_col(df, ["rating"])
    reviews_col = get_col(df, ["number_of_reviews"])

    if author_col is None or rating_col is None:
        st.error("Need 'author' and 'rating' columns.")
        return

    agg = {rating_col: ["mean", "count"]}
    if reviews_col:
        agg[reviews_col] = ["sum"]

    auth = df.groupby(author_col).agg(agg).round(2)
    cols = ["avg_rating", "book_count"]
    auth.columns = cols + (["total_reviews"] if reviews_col else [])
    auth = auth.reset_index()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_books = st.slider("Minimum books per author", 1, 10, 3)
    with col2:
        if reviews_col and "total_reviews" in auth.columns:
            pop_metric = st.radio(
                "Popularity metric",
                ["book_count", "total_reviews"],
                format_func=lambda x: "Book count" if x == "book_count" else "Total reviews",
            )
        else:
            pop_metric = "book_count"

    filtered = auth[auth["book_count"] >= min_books].copy()

    st.markdown(f"- Authors with ≥{min_books} books: `{len(filtered)}`")

    if filtered.empty:
        st.warning("No authors meet this filter; lower the minimum books.")
        return

    # Scatter popularity vs avg rating
    fig = px.scatter(
        filtered,
        x=pop_metric,
        y="avg_rating",
        hover_name=author_col,
        trendline="ols",
        title=f"Author Popularity vs Average Rating (metric: {pop_metric})",
        labels={pop_metric: pop_metric.replace("_", " ").title(), "avg_rating": "Average Rating"},
    )
    st.plotly_chart(fig, use_container_width=True)

    corr = filtered[[pop_metric, "avg_rating"]].corr().iloc[0, 1]
    st.metric("Correlation (popularity vs rating)", f"{corr:.4f}")

    with st.expander("Show author statistics table"):
        show_cols = [author_col, "avg_rating", "book_count"]
        if "total_reviews" in filtered.columns:
            show_cols.append("total_reviews")
        st.dataframe(filtered[show_cols].sort_values("avg_rating", ascending=False),
                     use_container_width=True, hide_index=True)

    st.markdown("#### Interpretation")
    strength = (
        "weak" if abs(corr) < 0.3 else
        "moderate" if abs(corr) < 0.7 else
        "strong"
    )
    st.info(
        f"""
        - Correlation of **{corr:.4f}** indicates a **{strength}** relationship  
          between author popularity and their average ratings.
        - You can use this to check whether popular authors are also consistently high‑rated.
        """
    )


# ===================================================================
# Q4 – Which feature combination gives most accurate predictions?
# ===================================================================

def show_q4(df: pd.DataFrame):
    st.markdown("## 4️⃣ Feature Combination Importance (Rating Prediction Proxy)")

    rating_col = get_col(df, ["rating"])
    genre_col  = get_col(df, ["genre"])
    author_col = get_col(df, ["author"])
    price_col  = get_col(df, ["price"])
    reviews_col = get_col(df, ["number_of_reviews"])
    year_col   = get_col(df, ["year"])

    if rating_col is None:
        st.error("Need 'rating' column.")
        return

    numeric_features = [c for c in [price_col, reviews_col, year_col] if c is not None]
    categorical_features = [c for c in [genre_col, author_col] if c is not None]

    st.markdown("**Available feature groups:**")
    st.write("- Numeric:", numeric_features if numeric_features else "None")
    st.write("- Categorical:", categorical_features if categorical_features else "None")

    col1, col2 = st.columns(2)
    with col1:
        use_num = st.checkbox("Include numeric features", value=bool(numeric_features))
    with col2:
        use_cat = st.checkbox("Include categorical features", value=bool(categorical_features))

    selected_numeric = numeric_features if use_num else []
    selected_categorical = categorical_features if use_cat else []

    if not selected_numeric and not selected_categorical:
        st.warning("Select at least one feature group.")
        return

    feature_cols = selected_numeric + selected_categorical

    model_df = df.dropna(subset=[rating_col]).copy()
    X = model_df[feature_cols]
    y = model_df[rating_col]

    transformers = []
    if selected_numeric:
        transformers.append(("num", "passthrough", selected_numeric))
    if selected_categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), selected_categorical))

    pre = ColumnTransformer(transformers=transformers)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    pipe = Pipeline([("pre", pre), ("rf", rf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with st.spinner("Training Random Forest model..."):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    st.metric("R² on hold-out set", f"{r2:.3f}")

    importances = rf.feature_importances_

    # Aggregate importance by original column
    rows = []
    if selected_numeric and selected_categorical:
        num_len = len(selected_numeric)
        # Numeric part
        for col, imp in zip(selected_numeric, importances[:num_len]):
            rows.append({"feature": col, "importance": imp})

        # Categorical: group one-hot back by original column
        ohe = pre.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(selected_categorical)
        cat_map = {c: 0.0 for c in selected_categorical}
        for name, imp in zip(cat_names, importances[num_len:]):
            orig = name.split("_")[0]  # 'genre_xxx' -> 'genre'
            cat_map[orig] += imp
        for col, imp in cat_map.items():
            rows.append({"feature": col, "importance": imp})

    elif selected_numeric:  # only numeric
        for col, imp in zip(selected_numeric, importances):
            rows.append({"feature": col, "importance": imp})

    else:  # only categorical
        ohe = pre.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(selected_categorical)
        cat_map = {c: 0.0 for c in selected_categorical}
        for name, imp in zip(cat_names, importances):
            orig = name.split("_")[0]
            cat_map[orig] += imp
        for col, imp in cat_map.items():
            rows.append({"feature": col, "importance": imp})

    imp_df = pd.DataFrame(rows)
    imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()
    imp_df = imp_df.sort_values("importance_norm", ascending=False)

    st.markdown("### Normalized feature importance (by original column)")
    fig = px.bar(
        imp_df,
        x="importance_norm",
        y="feature",
        orientation="h",
        labels={"importance_norm": "Normalized Importance", "feature": "Feature"},
        title="Relative Importance of Feature Groups for Predicting Ratings",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(imp_df, use_container_width=True, hide_index=True)

    st.markdown("#### Interpretation")
    st.info(
        """
        - Features with **higher normalized importance** contribute more to predicting ratings.  
        - You can treat this as a proxy for which feature combinations are most useful for
          building accurate recommendation models (e.g. genre + author vs price + reviews).  
        """
    )


# ===================================================================
# MAIN
# ===================================================================

def main():
    st.title("❓ Medium Analysis Questions")
    st.markdown(
        """
        This page explores **medium‑level questions** that combine multiple tables/models:

        1. **Which books are frequently clustered together based on descriptions?**  
        2. **How does genre similarity affect book recommendations?**  
        3. **What is the effect of author popularity on book ratings?**  
        4. **Which combination of features provides the most accurate recommendations?**
        """
    )

    df = load_data()
    if df.empty:
        st.error("No data available. Run the data pipeline and clustering first.")
        return

    with st.sidebar:
        st.header("📑 Select Question")
        q_choice = st.radio(
            "Choose analysis:",
            [
                "1️⃣ Clustered Books",
                "2️⃣ Genre Similarity",
                "3️⃣ Author Popularity vs Ratings",
                "4️⃣ Feature Importance",
                "📊 View All (1–4)",
            ],
        )

    if q_choice == "1️⃣ Clustered Books":
        show_q1(df)
    elif q_choice == "2️⃣ Genre Similarity":
        show_q2(df)
    elif q_choice == "3️⃣ Author Popularity vs Ratings":
        show_q3(df)
    elif q_choice == "4️⃣ Feature Importance":
        show_q4(df)
    else:  # View all
        show_q1(df)
        st.markdown("---")
        show_q2(df)
        st.markdown("---")
        show_q3(df)
        st.markdown("---")
        show_q4(df)


if __name__ == "__main__":
    main()
