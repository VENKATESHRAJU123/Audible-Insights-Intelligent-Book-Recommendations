"""
8_❓_Scenario_Recommendations.py

Scenario-Based Recommendations

1. A new user likes science fiction books. Which top 5 books should be recommended?
2. For a user who has previously rated thrillers highly, recommend similar books.
3. Identify books that are highly rated but have low popularity (hidden gems).
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px

# Make sure we can access project root if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Scenario Recommendations",
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


# -------------------------------------------------------------------
# Scenario 1 – Sci‑Fi lover, top 5 books
# -------------------------------------------------------------------
def show_scenario1(df: pd.DataFrame):
    st.markdown("## 1️⃣ New User Likes Science Fiction – Top 5 Books")

    genre_col   = get_col(df, ["genre"])
    rating_col  = get_col(df, ["rating"])
    reviews_col = get_col(df, ["number_of_reviews"])
    book_col    = get_col(df, ["book_name"])
    author_col  = get_col(df, ["author"])

    if not all([genre_col, rating_col, book_col]):
        st.error("Need 'genre', 'rating', and 'book_name' columns in clustered_data.csv.")
        return

    # Filters in sidebar
    with st.sidebar:
        st.markdown("### Scenario 1 Settings")
        min_rating = st.slider("Minimum rating", 0.0, 5.0, 4.0, 0.1)
        top_n = st.slider("Top N recommendations", 1, 20, 5)

    # Sci‑Fi / Science Fiction filter
    mask_scifi = df[genre_col].str.contains(
        "sci-fi|science fiction|science-fiction",
        case=False, na=False, regex=True
    )
    scifi_df = df[mask_scifi].copy()

    if scifi_df.empty:
        st.warning("No science fiction books found in the dataset.")
        return

    # Apply rating filter
    scifi_df = scifi_df[scifi_df[rating_col] >= min_rating]

    if scifi_df.empty:
        st.warning(f"No science fiction books with rating ≥ {min_rating}.")
        return

    # Sort: rating desc, reviews desc
    sort_cols = [rating_col]
    sort_asc  = [False]
    if reviews_col:
        sort_cols.append(reviews_col)
        sort_asc.append(False)

    scifi_sorted = scifi_df.sort_values(sort_cols, ascending=sort_asc)
    recs = scifi_sorted.head(top_n)

    display_cols = [book_col]
    for c in [author_col, genre_col, rating_col, reviews_col]:
        if c in df.columns:
            display_cols.append(c)

    st.markdown(f"### Recommended Top {len(recs)} Sci‑Fi Books")
    st.dataframe(recs[display_cols], use_container_width=True, hide_index=True)

    if not recs.empty:
        fig = px.bar(
            recs,
            x=rating_col,
            y=book_col,
            orientation="h",
            title="Sci‑Fi Recommendations (sorted by rating & popularity)",
            labels={rating_col: "Rating", book_col: "Book"},
            color=rating_col,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Explanation")
    st.info(
        f"""
        - Filtered books whose **genre contains 'Sci‑Fi' or 'Science Fiction'**  
        - Kept only books with **rating ≥ {min_rating}**  
        - Sorted by **rating** and then by **number of reviews** (popularity)  
        - Returned the top **{top_n}** as recommendations for a Sci‑Fi‑loving new user.
        """
    )


# -------------------------------------------------------------------
# Scenario 2 – Thriller lover, similar books
# -------------------------------------------------------------------
def show_scenario2(df: pd.DataFrame):
    st.markdown("## 2️⃣ User Loves Thrillers – Similar Book Recommendations")

    genre_col    = get_col(df, ["genre"])
    rating_col   = get_col(df, ["rating"])
    book_col     = get_col(df, ["book_name"])
    author_col   = get_col(df, ["author"])
    reviews_col  = get_col(df, ["number_of_reviews"])
    cluster_col  = get_col(df, ["cluster_kmeans", "cluster"])

    if not all([genre_col, rating_col, book_col, cluster_col]):
        st.error("Need 'genre', 'rating', 'book_name', and 'cluster_kmeans'/'cluster' columns.")
        return

    with st.sidebar:
        st.markdown("### Scenario 2 Settings")
        thr_rating_th = st.slider(
            "Min rating for user-liked thrillers",
            0.0, 5.0, 4.0, 0.1
        )
        rec_rating_th = st.slider(
            "Min rating for recommended books",
            0.0, 5.0, 4.0, 0.1
        )
        max_recs = st.slider("Max recommendations", 5, 50, 15)

    # 1. Thriller books
    is_thriller = df[genre_col].str.contains("thriller", case=False, na=False)
    thr_df = df[is_thriller].copy()

    if thr_df.empty:
        st.warning("No thriller books found in the dataset.")
        return

    # 2. High-rated thrillers (approx user-highly-rated)
    liked_thrillers = thr_df[thr_df[rating_col] >= thr_rating_th]
    if liked_thrillers.empty:
        st.warning(f"No thrillers with rating ≥ {thr_rating_th}.")
        return

    st.markdown("### User's Hypothetical 'Liked' Thrillers (basis for similarity)")
    liked_cols = [book_col, author_col, genre_col, rating_col]
    liked_cols = [c for c in liked_cols if c in liked_thrillers.columns]
    st.dataframe(
        liked_thrillers[liked_cols].sort_values(rating_col, ascending=False).head(10),
        use_container_width=True,
        hide_index=True
    )

    # 3. Get clusters containing those high-rated thrillers
    liked_clusters = liked_thrillers[cluster_col].unique().tolist()

    # Candidate recommendations: all books in those clusters, excluding the liked thrillers
    rec_candidates = df[df[cluster_col].isin(liked_clusters)].copy()
    rec_candidates = rec_candidates[~rec_candidates.index.isin(liked_thrillers.index)]

    # Filter by recommended rating threshold
    rec_candidates = rec_candidates[rec_candidates[rating_col] >= rec_rating_th]

    if rec_candidates.empty:
        st.warning("No suitable similar books found in thriller clusters with given filters.")
        return

    # Sort by rating then reviews
    sort_cols = [rating_col]
    sort_asc  = [False]
    if reviews_col:
        sort_cols.append(reviews_col)
        sort_asc.append(False)
    rec_sorted = rec_candidates.sort_values(sort_cols, ascending=sort_asc)

    recs = rec_sorted.head(max_recs)

    display_cols = [book_col]
    for c in [author_col, genre_col, rating_col, reviews_col, cluster_col]:
        if c in df.columns:
            display_cols.append(c)

    st.markdown(f"### Recommended Books (similar to high-rated thrillers) – Top {len(recs)}")
    st.dataframe(recs[display_cols], use_container_width=True, hide_index=True)

    st.markdown("#### Explanation")
    st.info(
        f"""
        - Found **thriller books** and kept those with rating ≥ `{thr_rating_th}`  
          as the *user's liked thrillers*.
        - Collected clusters that contain these high-rated thrillers (`{cluster_col}` labels).
        - From those clusters, recommended **other books** with rating ≥ `{rec_rating_th}`  
          (not necessarily only thrillers).
        - This approximates a **cluster-based, content‑similar recommendation** strategy.
        """
    )


# -------------------------------------------------------------------
# Scenario 3 – Hidden gems
# -------------------------------------------------------------------
def show_scenario3(df: pd.DataFrame):
    st.markdown("## 3️⃣ Hidden Gems – Highly Rated but Low Popularity")

    rating_col  = get_col(df, ["rating"])
    reviews_col = get_col(df, ["number_of_reviews"])
    book_col    = get_col(df, ["book_name"])
    author_col  = get_col(df, ["author"])
    genre_col   = get_col(df, ["genre"])

    if not all([rating_col, reviews_col, book_col]):
        st.error("Need 'rating', 'number_of_reviews', and 'book_name' columns.")
        return

    # Controls
    with st.sidebar:
        st.markdown("### Scenario 3 Settings")
        rating_th = st.slider("Min rating for hidden gems", 0.0, 5.0, 4.5, 0.1)
        max_reviews = st.slider("Max review count (popularity upper bound)", 1, 500, 50)
        max_gems = st.slider("Max hidden gems to list", 5, 100, 30)

    # Filter
    mask_high   = df[rating_col] >= rating_th
    mask_lowpop = df[reviews_col] <= max_reviews
    hidden_df = df[mask_high & mask_lowpop].copy()

    if hidden_df.empty:
        st.warning(f"No hidden gems with rating ≥ {rating_th} and reviews ≤ {max_reviews}.")
        return

    hidden_sorted = hidden_df.sort_values(
        [rating_col, reviews_col],
        ascending=[False, True]
    )
    gems = hidden_sorted.head(max_gems)

    display_cols = [book_col]
    for c in [author_col, genre_col, rating_col, reviews_col]:
        if c in df.columns:
            display_cols.append(c)

    st.markdown(f"### Top {len(gems)} Hidden Gems")
    st.dataframe(gems[display_cols], use_container_width=True, hide_index=True)

    # Quick visualization
    fig = px.scatter(
        gems,
        x=reviews_col,
        y=rating_col,
        hover_data=[book_col, author_col, genre_col] if all([book_col, author_col, genre_col]) else None,
        title="Hidden Gems: Rating vs Review Count",
        labels={reviews_col: "Number of Reviews", rating_col: "Rating"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Explanation")
    st.info(
        f"""
        - **Hidden gems** are books with **rating ≥ {rating_th}** but **reviews ≤ {max_reviews}**.  
        - They are likely **high-quality but not widely discovered**.  
        - Sorting by rating (descending) and review count (ascending) surfaces the best candidates.
        """
    )


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    st.title("❓ Scenario-Based Recommendations")

    st.markdown(
        """
        This page answers three scenario-based recommendation questions:

        1. **Sci‑Fi Lover** – Recommend top 5 science fiction books.  
        2. **Thriller Fan** – Recommend books similar to thrillers the user has rated highly.  
        3. **Hidden Gems** – Find highly rated but low‑popularity books.
        """
    )

    df = load_data()
    if df.empty:
        st.error("No data available. Make sure `data/processed/clustered_data.csv` exists.")
        return

    with st.sidebar:
        st.header("📑 Select Scenario")
        scenario = st.radio(
            "Choose a scenario:",
            [
                "1️⃣ Sci‑Fi Lover",
                "2️⃣ Thriller Fan – Similar Books",
                "3️⃣ Hidden Gems",
                "📊 View All Scenarios",
            ],
        )

    if scenario == "1️⃣ Sci‑Fi Lover":
        show_scenario1(df)
    elif scenario == "2️⃣ Thriller Fan – Similar Books":
        show_scenario2(df)
    elif scenario == "3️⃣ Hidden Gems":
        show_scenario3(df)
    else:  # View All
        show_scenario1(df)
        st.markdown("---")
        show_scenario2(df)
        st.markdown("---")
        show_scenario3(df)


if __name__ == "__main__":
    main()
