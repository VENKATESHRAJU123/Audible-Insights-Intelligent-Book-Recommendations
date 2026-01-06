"""
Clustering Module
Apply clustering algorithms to group similar books
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookClusterer:
    """Cluster books based on features"""
    
    def __init__(self, processed_data_path: str = "data/processed",
                 output_path: str = "outputs/visualizations"):
        """Initialize BookClusterer"""
        self.processed_data_path = Path(processed_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.kmeans_model = None
        self.dbscan_model = None
        
        logger.info("BookClusterer initialized")
    
    
    def load_feature_matrix(self) -> pd.DataFrame:
        """Load the feature matrix"""
        logger.info("Loading feature matrix...")
        
        file_path = self.processed_data_path / "feature_matrix.csv"
        df = pd.read_csv(file_path, encoding='utf-8')
        
        logger.info(f"Feature matrix loaded: {df.shape}")
        return df
    
    
    def prepare_clustering_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for clustering
        
        Args:
            df: Feature matrix
            
        Returns:
            Numpy array of features
        """
        logger.info("Preparing features for clustering...")
        
        # Select only numeric feature columns
        feature_cols = (
            [col for col in df.columns if col.startswith('tfidf_')] +
            [col for col in df.columns if col.endswith('_encoded')] +
            [col for col in df.columns if col.endswith('_scaled')]
        )
        
        X = df[feature_cols].values
        logger.info(f"Clustering features shape: {X.shape}")
        
        return X, feature_cols
    
    
    def find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method
        
        Args:
            X: Feature matrix
            max_k: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        logger.info("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'clustering_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Select optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    
    def apply_kmeans(self, X: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """
        Apply K-Means clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (if None, will find optimal)
            
        Returns:
            Cluster labels
        """
        logger.info("Applying K-Means clustering...")
        
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X)
        
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        labels = self.kmeans_model.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        logger.info(f"K-Means completed with {n_clusters} clusters")
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        
        return labels
    
    
    def apply_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Apply DBSCAN clustering
        
        Args:
            X: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Cluster labels
        """
        logger.info("Applying DBSCAN clustering...")
        
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan_model.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"DBSCAN completed:")
        logger.info(f"Number of clusters: {n_clusters}")
        logger.info(f"Number of noise points: {n_noise}")
        
        if n_clusters > 1:
            # Calculate silhouette score (excluding noise points)
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X[mask], labels[mask])
                logger.info(f"Silhouette Score: {silhouette:.4f}")
        
        return labels
    
    
    def visualize_clusters(self, df: pd.DataFrame, labels: np.ndarray, method: str = "kmeans"):
        """
        Visualize cluster distribution
        
        Args:
            df: Original dataframe
            labels: Cluster labels
            method: Clustering method name
        """
        logger.info(f"Visualizing {method} clusters...")
        
        # Cluster size distribution
        plt.figure(figsize=(10, 6))
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(unique, counts)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Books')
        plt.title(f'Cluster Distribution ({method.upper()})')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(self.output_path / f'{method}_cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster visualization saved")
    
    
    def create_clustered_data(self, df: pd.DataFrame, kmeans_labels: np.ndarray,
                            dbscan_labels: np.ndarray = None) -> pd.DataFrame:
        """
        Create final clustered dataset
        
        Args:
            df: Feature matrix
            kmeans_labels: K-Means cluster labels
            dbscan_labels: DBSCAN cluster labels (optional)
            
        Returns:
            Dataframe with cluster assignments
        """
        logger.info("Creating clustered dataset...")
        
        df_clustered = df.copy()
        df_clustered['cluster_kmeans'] = kmeans_labels
        
        if dbscan_labels is not None:
            df_clustered['cluster_dbscan'] = dbscan_labels
        
        # Save clustered data
        output_path = self.processed_data_path / "clustered_data.csv"
        df_clustered.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Clustered data saved: {df_clustered.shape}")
        logger.info(f"Saved to: {output_path}")
        
        return df_clustered
    
    
    def analyze_clusters(self, df: pd.DataFrame):
        """
        Analyze cluster characteristics
        
        Args:
            df: Clustered dataframe
        """
        logger.info("Analyzing cluster characteristics...")
        
        if 'cluster_kmeans' not in df.columns:
            logger.warning("No cluster labels found")
            return
        
        # Analyze by cluster
        for cluster_id in df['cluster_kmeans'].unique():
            cluster_df = df[df['cluster_kmeans'] == cluster_id]
            
            print(f"\n{'='*50}")
            print(f"CLUSTER {cluster_id}")
            print(f"{'='*50}")
            print(f"Size: {len(cluster_df)} books")
            
            if 'rating' in df.columns:
                print(f"Average Rating: {cluster_df['rating'].mean():.2f}")
            
            if 'genre' in df.columns:
                top_genres = cluster_df['genre'].value_counts().head(3)
                print(f"Top Genres:\n{top_genres}")
            
            if 'author' in df.columns:
                top_authors = cluster_df['author'].value_counts().head(3)
                print(f"Top Authors:\n{top_authors}")


def main():
    """Main execution function"""
    
    # Initialize clusterer
    clusterer = BookClusterer()
    
    # Load feature matrix
    df = clusterer.load_feature_matrix()
    
    # Prepare features
    X, feature_cols = clusterer.prepare_clustering_features(df)
    
    # Apply K-Means
    kmeans_labels = clusterer.apply_kmeans(X)
    clusterer.visualize_clusters(df, kmeans_labels, method="kmeans")
    
    # Apply DBSCAN (optional)
    dbscan_labels = clusterer.apply_dbscan(X, eps=1.5, min_samples=5)
    clusterer.visualize_clusters(df, dbscan_labels, method="dbscan")
    
    # Create clustered dataset
    df_clustered = clusterer.create_clustered_data(df, kmeans_labels, dbscan_labels)
    
    # Analyze clusters
    clusterer.analyze_clusters(df_clustered)
    
    print("\n" + "="*50)
    print("CLUSTERING COMPLETE")
    print("="*50)
    
    logger.info("\n✅ Clustering completed successfully!")


if __name__ == "__main__":
    main()
