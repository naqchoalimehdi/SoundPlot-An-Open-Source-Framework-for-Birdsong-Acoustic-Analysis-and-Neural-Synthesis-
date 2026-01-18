"""
Data export utilities.

Handles exporting analysis results to various formats
for storage and visualization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class DataExporter:
    """
    Export analysis results to various formats.
    
    Supports Parquet (efficient), JSON (readable), and CSV.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "data/results"):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _serialize_numpy(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy(v) for v in obj]
        return obj
    
    def export_features(
        self,
        features: List[Dict],
        segment_info: Optional[List[Dict]] = None,
        filename: str = "features"
    ) -> Path:
        """
        Export extracted features to Parquet.
        
        Args:
            features: List of feature dictionaries.
            segment_info: Optional list of segment metadata.
            filename: Output filename (without extension).
            
        Returns:
            Path to the exported file.
        """
        df = pd.DataFrame(features)
        
        if segment_info:
            info_df = pd.DataFrame(segment_info)
            df = pd.concat([info_df, df], axis=1)
        
        output_path = self.output_dir / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        
        return output_path
    
    def export_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
        filename: str = "embeddings"
    ) -> Path:
        """
        Export dimensionality-reduced embeddings.
        
        Args:
            embeddings: 2D array of shape (n_samples, n_components).
            labels: Optional cluster labels.
            metadata: Optional metadata per sample.
            filename: Output filename.
            
        Returns:
            Path to the exported file.
        """
        n_dims = embeddings.shape[1]
        
        data = {}
        for i in range(n_dims):
            data[f"dim_{i}"] = embeddings[:, i]
        
        if labels is not None:
            data["cluster"] = labels
        
        df = pd.DataFrame(data)
        
        if metadata:
            meta_df = pd.DataFrame(metadata)
            df = pd.concat([meta_df, df], axis=1)
        
        output_path = self.output_dir / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        
        return output_path
    
    def export_clustering_results(
        self,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        statistics: Optional[Dict] = None,
        filename: str = "clustering"
    ) -> Path:
        """
        Export clustering results.
        
        Args:
            labels: Cluster labels.
            probabilities: Cluster membership probabilities.
            statistics: Clustering statistics.
            filename: Output filename.
            
        Returns:
            Path to the exported file.
        """
        data = {"label": labels}
        
        if probabilities is not None:
            data["probability"] = probabilities
        
        df = pd.DataFrame(data)
        
        # Export data
        data_path = self.output_dir / f"{filename}_data.parquet"
        df.to_parquet(data_path, index=False)
        
        # Export statistics if provided
        if statistics:
            stats_path = self.output_dir / f"{filename}_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self._serialize_numpy(statistics), f, indent=2)
        
        return data_path
    
    def export_patterns(
        self,
        patterns: Dict,
        filename: str = "patterns"
    ) -> Path:
        """
        Export pattern analysis results.
        
        Args:
            patterns: Pattern analysis dictionary.
            filename: Output filename.
            
        Returns:
            Path to the exported file.
        """
        output_path = self.output_dir / f"{filename}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self._serialize_numpy(patterns), f, indent=2)
        
        return output_path
    
    def export_full_analysis(
        self,
        features: np.ndarray,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        statistics: Optional[Dict] = None,
        patterns: Optional[Dict] = None,
        filename: str = "full_analysis"
    ) -> Dict[str, Path]:
        """
        Export complete analysis results.
        
        Args:
            features: Feature matrix.
            embeddings: Reduced embeddings.
            labels: Cluster labels.
            metadata: Per-sample metadata.
            statistics: Analysis statistics.
            patterns: Pattern analysis.
            filename: Base filename.
            
        Returns:
            Dictionary mapping result type to file path.
        """
        paths = {}
        
        # Create combined dataframe
        n_features = features.shape[1]
        n_dims = embeddings.shape[1]
        
        data = {}
        
        # Add metadata
        if metadata:
            for i, meta in enumerate(metadata):
                for key, value in meta.items():
                    if key not in data:
                        data[key] = []
                    data[key].append(value)
        
        # Add features
        for i in range(n_features):
            data[f"feature_{i}"] = features[:, i]
        
        # Add embeddings
        for i in range(n_dims):
            data[f"embedding_{i}"] = embeddings[:, i]
        
        # Add labels
        data["cluster"] = labels
        
        df = pd.DataFrame(data)
        
        # Export main data
        main_path = self.output_dir / f"{filename}.parquet"
        df.to_parquet(main_path, index=False)
        paths["data"] = main_path
        
        # Also export to CSV for compatibility
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        paths["csv"] = csv_path
        
        # Export statistics
        if statistics:
            stats_path = self.output_dir / f"{filename}_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self._serialize_numpy(statistics), f, indent=2)
            paths["statistics"] = stats_path
        
        # Export patterns
        if patterns:
            patterns_path = self.output_dir / f"{filename}_patterns.json"
            with open(patterns_path, 'w') as f:
                json.dump(self._serialize_numpy(patterns), f, indent=2)
            paths["patterns"] = patterns_path
        
        return paths
    
    def load_results(
        self,
        filename: str
    ) -> pd.DataFrame:
        """
        Load previously exported results.
        
        Args:
            filename: Filename (with or without extension).
            
        Returns:
            DataFrame with results.
        """
        # Try parquet first
        parquet_path = self.output_dir / f"{filename}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        
        # Try without extension
        if (self.output_dir / filename).exists():
            return pd.read_parquet(self.output_dir / filename)
        
        # Try CSV
        csv_path = self.output_dir / f"{filename}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        
        raise FileNotFoundError(f"Results file not found: {filename}")
