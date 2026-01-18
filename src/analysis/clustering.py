"""
Clustering analysis for audio features.

Provides HDBSCAN for automatic cluster discovery and
K-Means for when a specific number of clusters is desired.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class ClusterAnalyzer:
    """
    Cluster audio features to discover patterns.
    
    Uses HDBSCAN for automatic cluster discovery or
    K-Means when the number of clusters is known.
    """
    
    def __init__(
        self,
        method: str = "hdbscan",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        random_state: int = 42
    ):
        """
        Initialize cluster analyzer.
        
        Args:
            method: Clustering method ("hdbscan" or "kmeans").
            n_clusters: Number of clusters (required for kmeans).
            min_cluster_size: Minimum cluster size (HDBSCAN).
            min_samples: Minimum samples for core points (HDBSCAN).
            random_state: Random seed for reproducibility.
        """
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.random_state = random_state
        
        self.clusterer = None
        self.labels_ = None
        self.probabilities_ = None
        
        if self.method == "hdbscan" and not HDBSCAN_AVAILABLE:
            print("Warning: HDBSCAN not available, falling back to K-Means")
            self.method = "kmeans"
            if self.n_clusters is None:
                self.n_clusters = 5  # Default
    
    def fit(self, features: np.ndarray) -> 'ClusterAnalyzer':
        """
        Fit the clusterer to features.
        
        Args:
            features: 2D array of shape (n_samples, n_features).
            
        Returns:
            Self for chaining.
        """
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.method == "hdbscan":
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=0.0,
                metric='euclidean'
            )
            self.labels_ = self.clusterer.fit_predict(features)
            self.probabilities_ = self.clusterer.probabilities_
        else:
            if self.n_clusters is None:
                raise ValueError("n_clusters required for K-Means")
            
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            self.labels_ = self.clusterer.fit_predict(features)
            
            # Compute soft assignment probabilities
            distances = self.clusterer.transform(features)
            self.probabilities_ = 1.0 / (1.0 + distances.min(axis=1))
        
        return self
    
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and return cluster labels.
        
        Args:
            features: 2D array of shape (n_samples, n_features).
            
        Returns:
            1D array of cluster labels (-1 for noise in HDBSCAN).
        """
        self.fit(features)
        return self.labels_
    
    def get_cluster_statistics(
        self,
        features: np.ndarray
    ) -> Dict[str, Union[int, float]]:
        """
        Compute clustering statistics.
        
        Args:
            features: Feature array used for clustering.
            
        Returns:
            Dictionary with clustering metrics.
        """
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        unique_labels = set(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.labels_ == -1)
        
        stats = {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio": n_noise / len(self.labels_) if len(self.labels_) > 0 else 0,
        }
        
        # Compute cluster sizes
        for label in unique_labels:
            if label == -1:
                continue
            size = np.sum(self.labels_ == label)
            stats[f"cluster_{label}_size"] = size
        
        # Silhouette score (only if more than 1 cluster and not all noise)
        valid_mask = self.labels_ != -1
        if n_clusters > 1 and np.sum(valid_mask) > n_clusters:
            try:
                stats["silhouette_score"] = float(silhouette_score(
                    features[valid_mask],
                    self.labels_[valid_mask]
                ))
            except Exception:
                stats["silhouette_score"] = 0.0
            
            try:
                stats["calinski_harabasz_score"] = float(calinski_harabasz_score(
                    features[valid_mask],
                    self.labels_[valid_mask]
                ))
            except Exception:
                stats["calinski_harabasz_score"] = 0.0
        
        return stats
    
    def get_cluster_centers(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Compute cluster centers (centroids).
        
        Args:
            features: Feature array.
            
        Returns:
            2D array of shape (n_clusters, n_features).
        """
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        if self.method == "kmeans":
            return self.clusterer.cluster_centers_
        
        # For HDBSCAN, compute centroids manually
        unique_labels = set(self.labels_) - {-1}
        centers = []
        
        for label in sorted(unique_labels):
            mask = self.labels_ == label
            center = np.mean(features[mask], axis=0)
            centers.append(center)
        
        return np.array(centers)
    
    def find_exemplars(
        self,
        features: np.ndarray,
        n_exemplars: int = 3
    ) -> Dict[int, List[int]]:
        """
        Find exemplar (most typical) points per cluster.
        
        Args:
            features: Feature array.
            n_exemplars: Number of exemplars per cluster.
            
        Returns:
            Dictionary mapping cluster ID to list of exemplar indices.
        """
        if self.labels_ is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        centers = self.get_cluster_centers(features)
        unique_labels = sorted(set(self.labels_) - {-1})
        
        exemplars = {}
        
        for i, label in enumerate(unique_labels):
            mask = self.labels_ == label
            indices = np.where(mask)[0]
            cluster_features = features[mask]
            
            # Find points closest to center
            center = centers[i]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            
            # Get top n_exemplars
            n = min(n_exemplars, len(indices))
            closest_in_cluster = np.argsort(distances)[:n]
            exemplars[label] = [int(indices[i]) for i in closest_in_cluster]
        
        return exemplars


def find_optimal_clusters(
    features: np.ndarray,
    max_clusters: int = 10,
    random_state: int = 42
) -> Tuple[int, Dict[int, float]]:
    """
    Find optimal number of clusters using elbow method.
    
    Args:
        features: Feature array.
        max_clusters: Maximum clusters to try.
        random_state: Random seed.
        
    Returns:
        Tuple of (optimal_k, {k: silhouette_score}).
    """
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    scores = {}
    
    for k in range(2, min(max_clusters + 1, len(features))):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features)
        
        try:
            score = silhouette_score(features, labels)
            scores[k] = score
        except Exception:
            scores[k] = 0.0
    
    if not scores:
        return 2, {}
    
    optimal_k = max(scores, key=scores.get)
    
    return optimal_k, scores
