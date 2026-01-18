"""
Visualization utilities for verification.

Provides static plots for inspecting features,
embeddings, and clustering results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Visualizer:
    """
    Create static visualizations for analysis verification.
    
    These are meant for quick inspection; the full web UI
    will provide interactive visualizations.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "data/plots",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving plots.
            figsize: Default figure size.
            dpi: Resolution for saved figures.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int,
        title: str = "Spectrogram",
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot audio spectrogram.
        
        Args:
            audio: Audio signal.
            sample_rate: Sample rate.
            title: Plot title.
            filename: If provided, save to this file.
            
        Returns:
            Path to saved file if filename provided.
        """
        import librosa
        import librosa.display
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio)),
            ref=np.max
        )
        
        librosa.display.specshow(
            D,
            sr=sample_rate,
            x_axis='time',
            y_axis='log',
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        
        plt.colorbar(ax.images[0], ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        
        if filename:
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return path
        
        plt.show()
        return None
    
    def plot_embeddings_2d(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "2D Embedding",
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot 2D embeddings scatter plot.
        
        Args:
            embeddings: 2D array with at least 2 columns.
            labels: Optional cluster labels for coloring.
            title: Plot title.
            filename: If provided, save to this file.
            
        Returns:
            Path to saved file if filename provided.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        
        if labels is not None:
            # Handle noise label (-1)
            unique_labels = sorted(set(labels))
            n_colors = len(unique_labels)
            colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_colors))
            label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
            
            for label in unique_labels:
                mask = labels == label
                color = label_to_color[label]
                name = 'Noise' if label == -1 else f'Cluster {label}'
                ax.scatter(
                    x[mask], y[mask],
                    c=[color],
                    label=name,
                    alpha=0.6,
                    s=30
                )
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(x, y, alpha=0.6, s=30)
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if filename:
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return path
        
        plt.show()
        return None
    
    def plot_embeddings_3d(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "3D Embedding",
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot 3D embeddings scatter plot.
        
        Args:
            embeddings: 2D array with at least 3 columns.
            labels: Optional cluster labels for coloring.
            title: Plot title.
            filename: If provided, save to this file.
            
        Returns:
            Path to saved file if filename provided.
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        z = embeddings[:, 2] if embeddings.shape[1] > 2 else np.zeros_like(x)
        
        if labels is not None:
            unique_labels = sorted(set(labels))
            n_colors = len(unique_labels)
            colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_colors))
            label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
            
            for label in unique_labels:
                mask = labels == label
                color = label_to_color[label]
                name = 'Noise' if label == -1 else f'Cluster {label}'
                ax.scatter(
                    x[mask], y[mask], z[mask],
                    c=[color],
                    label=name,
                    alpha=0.6,
                    s=30
                )
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(x, y, z, alpha=0.6, s=30)
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if filename:
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return path
        
        plt.show()
        return None
    
    def plot_cluster_distribution(
        self,
        labels: np.ndarray,
        title: str = "Cluster Distribution",
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot histogram of cluster sizes.
        
        Args:
            labels: Cluster labels.
            title: Plot title.
            filename: If provided, save to this file.
            
        Returns:
            Path to saved file if filename provided.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Separate noise
        names = [f'C{l}' if l >= 0 else 'Noise' for l in unique_labels]
        colors = ['gray' if l < 0 else plt.cm.tab10(l % 10) for l in unique_labels]
        
        bars = ax.bar(names, counts, color=colors)
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.set_title(title)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        
        if filename:
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return path
        
        plt.show()
        return None
    
    def plot_transition_matrix(
        self,
        matrix: np.ndarray,
        labels: List[int],
        title: str = "Transition Matrix",
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot transition matrix heatmap.
        
        Args:
            matrix: 2D transition matrix.
            labels: State labels for axes.
            title: Plot title.
            filename: If provided, save to this file.
            
        Returns:
            Path to saved file if filename provided.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(matrix, cmap='Blues')
        
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([f'C{l}' for l in labels])
        ax.set_yticklabels([f'C{l}' for l in labels])
        
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix[i, j]
                if val > 0.01:
                    text = ax.text(
                        j, i, f'{val:.2f}',
                        ha='center', va='center',
                        color='white' if val > 0.5 else 'black'
                    )
        
        plt.tight_layout()
        
        if filename:
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return path
        
        plt.show()
        return None
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = "Feature Importance",
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot feature importance bar chart.
        
        Args:
            feature_names: Names of features.
            importances: Importance values.
            title: Plot title.
            filename: If provided, save to this file.
            
        Returns:
            Path to saved file if filename provided.
        """
        fig, ax = plt.subplots(figsize=(10, len(feature_names) * 0.3 + 2))
        
        # Sort by importance
        sorted_idx = np.argsort(importances)
        
        ax.barh(
            range(len(feature_names)),
            importances[sorted_idx],
            color='steelblue'
        )
        
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if filename:
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return path
        
        plt.show()
        return None
