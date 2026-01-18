"""
Unified feature extraction pipeline.

Combines all feature types (spectral, temporal, pitch, MFCC)
into a single interface for comprehensive audio analysis.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from .spectral import SpectralFeatures
from .temporal import TemporalFeatures
from .pitch import PitchFeatures
from .mfcc import MFCCFeatures


class FeatureExtractor:
    """
    Unified feature extractor combining all feature types.
    
    Provides a high-level interface for extracting a comprehensive
    set of acoustic features from audio signals.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
        pitch_fmin: float = 100.0,
        pitch_fmax: float = 8000.0
    ):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Audio sample rate.
            n_fft: FFT window size.
            hop_length: Hop length for analysis.
            n_mfcc: Number of MFCCs to extract.
            pitch_fmin: Minimum pitch frequency (Hz).
            pitch_fmax: Maximum pitch frequency (Hz).
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Initialize sub-extractors
        self.spectral = SpectralFeatures(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        self.temporal = TemporalFeatures(
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        
        self.pitch = PitchFeatures(
            sample_rate=sample_rate,
            fmin=pitch_fmin,
            fmax=pitch_fmax,
            hop_length=hop_length
        )
        
        self.mfcc = MFCCFeatures(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    def extract_all(
        self,
        audio: np.ndarray,
        include_spectral: bool = True,
        include_temporal: bool = True,
        include_pitch: bool = True,
        include_mfcc: bool = True
    ) -> Dict[str, float]:
        """
        Extract all features from audio.
        
        Args:
            audio: Input audio signal.
            include_spectral: Include spectral features.
            include_temporal: Include temporal features.
            include_pitch: Include pitch features.
            include_mfcc: Include MFCC features.
            
        Returns:
            Dictionary containing all requested features.
        """
        features = {}
        
        if include_spectral:
            features.update(self.spectral.extract_all(audio))
        
        if include_temporal:
            features.update(self.temporal.extract_all(audio))
        
        if include_pitch:
            features.update(self.pitch.extract_all(audio))
        
        if include_mfcc:
            features.update(self.mfcc.extract_all(audio))
        
        return features
    
    def extract_compact(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract a compact set of key features.
        
        Returns fewer features for faster processing and
        easier visualization.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with compact feature set.
        """
        features = {}
        
        # Spectral summary
        spectral = self.spectral.spectral_centroid(audio)
        features["spectral_centroid"] = spectral["spectral_centroid_mean"]
        
        bandwidth = self.spectral.spectral_bandwidth(audio)
        features["spectral_bandwidth"] = bandwidth["spectral_bandwidth_mean"]
        
        zcr = self.spectral.zero_crossing_rate(audio)
        features["zero_crossing_rate"] = zcr["zero_crossing_rate_mean"]
        
        # Temporal summary
        onsets = self.temporal.onset_detection(audio)
        features["onset_rate"] = onsets["onset_rate"]
        
        tempo = self.temporal.tempo_estimation(audio)
        features["tempo"] = tempo["tempo_bpm"]
        
        energy = self.temporal.energy_dynamics(audio)
        features["rms_mean"] = energy["rms_mean"]
        features["dynamic_range"] = energy["dynamic_range_db"]
        
        # Pitch summary
        pitch = self.pitch.pitch_statistics(audio)
        features["pitch_mean"] = pitch["pitch_mean_hz"]
        features["pitch_range"] = pitch["pitch_range_hz"]
        features["voiced_fraction"] = pitch["voiced_fraction"]
        
        # MFCC summary (first 5 coefficients)
        mfcc_stats = self.mfcc.mfcc_statistics(audio)
        for i in range(5):
            features[f"mfcc_{i}"] = mfcc_stats[f"mfcc_{i}_mean"]
        
        return features
    
    def extract_vector(
        self,
        audio: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract a fixed-length feature vector.
        
        Useful for machine learning where consistent
        input dimensions are required.
        
        Args:
            audio: Input audio signal.
            normalize: Whether to normalize the vector.
            
        Returns:
            1D numpy array of features.
        """
        features = self.extract_compact(audio)
        vector = np.array(list(features.values()), dtype=np.float32)
        
        # Replace NaN/inf with 0
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        if normalize:
            # Z-score normalization
            mean = np.mean(vector)
            std = np.std(vector)
            if std > 0:
                vector = (vector - mean) / std
        
        return vector
    
    def get_feature_names(self, compact: bool = True) -> List[str]:
        """
        Get the names of features in order.
        
        Args:
            compact: If True, return compact feature names.
            
        Returns:
            List of feature names.
        """
        if compact:
            return [
                "spectral_centroid",
                "spectral_bandwidth",
                "zero_crossing_rate",
                "onset_rate",
                "tempo",
                "rms_mean",
                "dynamic_range",
                "pitch_mean",
                "pitch_range",
                "voiced_fraction",
                "mfcc_0",
                "mfcc_1",
                "mfcc_2",
                "mfcc_3",
                "mfcc_4",
            ]
        else:
            # Return all feature names
            dummy = np.zeros(22050)  # 1 second of silence
            features = self.extract_all(dummy)
            return list(features.keys())
    
    def extract_batch(
        self,
        audio_list: List[np.ndarray],
        compact: bool = True
    ) -> np.ndarray:
        """
        Extract features from multiple audio segments.
        
        Args:
            audio_list: List of audio arrays.
            compact: Use compact feature set.
            
        Returns:
            2D array of shape (n_samples, n_features).
        """
        vectors = []
        
        for audio in audio_list:
            if compact:
                features = self.extract_compact(audio)
                vec = np.array(list(features.values()), dtype=np.float32)
            else:
                vec = self.extract_vector(audio, normalize=False)
            
            vectors.append(vec)
        
        return np.array(vectors, dtype=np.float32)
