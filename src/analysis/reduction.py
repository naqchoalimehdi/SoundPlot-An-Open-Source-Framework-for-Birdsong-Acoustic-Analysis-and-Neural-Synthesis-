"""
Dimensionality reduction for audio feature visualization.

Provides UMAP and PCA for projecting high-dimensional
feature vectors into 2D/3D for visualization.
"""

from typing import Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DimensionalityReducer:
    """
    Reduce high-dimensional feature vectors to 2D or 3D.
    
    Uses UMAP (preferred) or PCA for dimensionality reduction.
    UMAP preserves both local and global structure better than t-SNE.
    """
    
    def __init__(
        self,
        n_components: int = 3,
        method: str = "umap",
        random_state: int = 42
    ):
        """
        Initialize dimensionality reducer.
        
        Args:
            n_components: Number of dimensions (2 or 3).
            method: Reduction method ("umap" or "pca").
            random_state: Random seed for reproducibility.
        """
        self.n_components = n_components
        self.method = method.lower()
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.reducer = None
        self._is_fitted = False
        
        if self.method == "umap" and not UMAP_AVAILABLE:
            print("Warning: UMAP not available, falling back to PCA")
            self.method = "pca"
    
    def _create_reducer(self):
        """Create the reducer model."""
        if self.method == "umap":
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=self.random_state
            )
        else:
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
    
    def fit(self, features: np.ndarray) -> 'DimensionalityReducer':
        """
        Fit the reducer to features.
        
        Args:
            features: 2D array of shape (n_samples, n_features).
            
        Returns:
            Self for chaining.
        """
        # Handle NaN/inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create and fit reducer
        self._create_reducer()
        self.reducer.fit(features_scaled)
        self._is_fitted = True
        
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features to lower dimensions.
        
        Args:
            features: 2D array of shape (n_samples, n_features).
            
        Returns:
            2D array of shape (n_samples, n_components).
        """
        if not self._is_fitted:
            raise ValueError("Reducer not fitted. Call fit() first.")
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = self.scaler.transform(features)
        
        return self.reducer.transform(features_scaled)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            features: 2D array of shape (n_samples, n_features).
            
        Returns:
            2D array of shape (n_samples, n_components).
        """
        self.fit(features)
        
        # For UMAP, use the fitted embedding directly
        if self.method == "umap":
            return self.reducer.embedding_
        else:
            return self.transform(features)
    
    def get_explained_variance(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio (PCA only).
        
        Returns:
            Array of explained variance ratios, or None if UMAP.
        """
        if self.method == "pca" and self._is_fitted:
            return self.reducer.explained_variance_ratio_
        return None


class FeatureScaler:
    """
    Utility class for feature scaling/normalization.
    """
    
    def __init__(self, method: str = "standard"):
        """
        Initialize scaler.
        
        Args:
            method: Scaling method ("standard", "minmax", "robust").
        """
        self.method = method
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform features."""
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.fit_transform(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(features)
