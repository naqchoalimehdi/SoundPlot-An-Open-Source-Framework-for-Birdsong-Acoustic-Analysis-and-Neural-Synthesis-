"""
Comparison visualization for original vs synthesized audio.

Provides side-by-side visual comparisons of spectrograms,
waveforms, and feature embeddings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ComparisonVisualizer:
    """
    Visualize comparisons between original and synthesized audio.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "data/comparisons",
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 100
    ):
        """
        Initialize comparison visualizer.
        
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
    
    def plot_full_comparison(
        self,
        comparison_features: Dict[str, np.ndarray],
        metrics: Dict[str, float],
        sample_rate: int = 22050,
        title: str = "Original vs Synthesized",
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Create comprehensive comparison visualization.
        
        Shows:
        - Waveforms (original and synthesized)
        - Spectrograms (original and synthesized)
        - Difference spectrogram
        - Metrics summary
        
        Args:
            comparison_features: From SynthesisComparator.extract_comparison_features()
            metrics: From SynthesisComparator.compute_comparison_metrics()
            sample_rate: Audio sample rate.
            title: Main title.
            filename: If provided, save to this file.
            
        Returns:
            Path to saved file if filename provided.
        """
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1.5, 1.5, 0.5])
        
        # Waveforms
        ax_wave_orig = fig.add_subplot(gs[0, 0])
        ax_wave_synth = fig.add_subplot(gs[0, 1])
        
        orig_wave = comparison_features["original_waveform"]
        synth_wave = comparison_features["synthesized_waveform"]
        
        time_orig = np.arange(len(orig_wave)) / sample_rate
        time_synth = np.arange(len(synth_wave)) / sample_rate
        
        ax_wave_orig.plot(time_orig, orig_wave, color='#2196F3', linewidth=0.5)
        ax_wave_orig.set_title("Original Waveform")
        ax_wave_orig.set_xlabel("Time (s)")
        ax_wave_orig.set_ylabel("Amplitude")
        ax_wave_orig.set_ylim(-1, 1)
        
        ax_wave_synth.plot(time_synth, synth_wave, color='#4CAF50', linewidth=0.5)
        ax_wave_synth.set_title("Synthesized Waveform")
        ax_wave_synth.set_xlabel("Time (s)")
        ax_wave_synth.set_ylabel("Amplitude")
        ax_wave_synth.set_ylim(-1, 1)
        
        # Spectrograms
        ax_spec_orig = fig.add_subplot(gs[1, 0])
        ax_spec_synth = fig.add_subplot(gs[1, 1])
        
        import librosa.display
        
        librosa.display.specshow(
            comparison_features["original_spectrogram"],
            sr=sample_rate,
            x_axis='time',
            y_axis='log',
            ax=ax_spec_orig,
            cmap='magma'
        )
        ax_spec_orig.set_title("Original Spectrogram")
        
        librosa.display.specshow(
            comparison_features["synthesized_spectrogram"],
            sr=sample_rate,
            x_axis='time',
            y_axis='log',
            ax=ax_spec_synth,
            cmap='magma'
        )
        ax_spec_synth.set_title("Synthesized Spectrogram")
        
        # Mel spectrograms
        ax_mel_orig = fig.add_subplot(gs[2, 0])
        ax_mel_synth = fig.add_subplot(gs[2, 1])
        
        librosa.display.specshow(
            comparison_features["original_mel"],
            sr=sample_rate,
            x_axis='time',
            y_axis='mel',
            ax=ax_mel_orig,
            cmap='viridis'
        )
        ax_mel_orig.set_title("Original Mel Spectrogram")
        
        librosa.display.specshow(
            comparison_features["synthesized_mel"],
            sr=sample_rate,
            x_axis='time',
            y_axis='mel',
            ax=ax_mel_synth,
            cmap='viridis'
        )
        ax_mel_synth.set_title("Synthesized Mel Spectrogram")
        
        # Metrics
        ax_metrics = fig.add_subplot(gs[3, :])
        ax_metrics.axis('off')
        
        metrics_text = (
            f"SNR: {metrics['snr_db']:.2f} dB  |  "
            f"Waveform Corr: {metrics['waveform_correlation']:.3f}  |  "
            f"Spectral Corr: {metrics['spectral_correlation']:.3f}  |  "
            f"Mel Corr: {metrics['mel_correlation']:.3f}"
        )
        
        ax_metrics.text(
            0.5, 0.5, metrics_text,
            ha='center', va='center',
            fontsize=12,
            transform=ax_metrics.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return path
        
        plt.show()
        return None
    
    def plot_embedding_comparison(
        self,
        original_embeddings: np.ndarray,
        synthesized_embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Feature Space Comparison",
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Compare embeddings from original and synthesized audio.
        
        Shows how similar the features are in the reduced space.
        
        Args:
            original_embeddings: Embeddings from original audio.
            synthesized_embeddings: Embeddings from synthesized audio.
            labels: Optional cluster labels.
            title: Plot title.
            filename: Save filename.
            
        Returns:
            Path to saved file.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original embeddings
        ax = axes[0]
        if labels is not None:
            scatter = ax.scatter(
                original_embeddings[:, 0],
                original_embeddings[:, 1],
                c=labels, cmap='tab10', alpha=0.6
            )
        else:
            ax.scatter(
                original_embeddings[:, 0],
                original_embeddings[:, 1],
                alpha=0.6, color='#2196F3'
            )
        ax.set_title("Original Audio Features")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        
        # Synthesized embeddings
        ax = axes[1]
        if labels is not None:
            ax.scatter(
                synthesized_embeddings[:, 0],
                synthesized_embeddings[:, 1],
                c=labels, cmap='tab10', alpha=0.6
            )
        else:
            ax.scatter(
                synthesized_embeddings[:, 0],
                synthesized_embeddings[:, 1],
                alpha=0.6, color='#4CAF50'
            )
        ax.set_title("Synthesized Audio Features")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        
        # Overlay comparison
        ax = axes[2]
        ax.scatter(
            original_embeddings[:, 0],
            original_embeddings[:, 1],
            alpha=0.5, color='#2196F3', label='Original', s=50
        )
        ax.scatter(
            synthesized_embeddings[:, 0],
            synthesized_embeddings[:, 1],
            alpha=0.5, color='#4CAF50', label='Synthesized', s=50, marker='x'
        )
        
        # Draw lines connecting corresponding points
        for i in range(min(len(original_embeddings), len(synthesized_embeddings))):
            ax.plot(
                [original_embeddings[i, 0], synthesized_embeddings[i, 0]],
                [original_embeddings[i, 1], synthesized_embeddings[i, 1]],
                color='gray', alpha=0.3, linewidth=1
            )
        
        ax.set_title("Overlay (lines = drift)")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend()
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            path = self.output_dir / f"{filename}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return path
        
        plt.show()
        return None
