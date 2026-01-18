"""
Spectral feature extraction for audio analysis.

Extracts frequency-domain features that characterize
the timbral and harmonic content of audio signals.
"""

from typing import Dict, Optional
import numpy as np
import librosa


class SpectralFeatures:
    """
    Extract spectral features from audio signals.
    
    Features include spectral centroid, bandwidth, rolloff,
    contrast, flatness, and chroma features.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize spectral feature extractor.
        
        Args:
            sample_rate: Audio sample rate.
            n_fft: FFT window size.
            hop_length: Number of samples between frames.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def spectral_centroid(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral centroid (center of mass of spectrum).
        
        The spectral centroid indicates where the "center of mass"
        of the spectrum is located. Higher values = brighter sound.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with mean, std, min, max of centroid.
        """
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_centroid_min": float(np.min(centroid)),
            "spectral_centroid_max": float(np.max(centroid)),
        }
    
    def spectral_bandwidth(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral bandwidth (spread around centroid).
        
        Indicates the width of the spectral distribution.
        Higher values = more complex/rich harmonics.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with bandwidth statistics.
        """
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        return {
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_bandwidth_std": float(np.std(bandwidth)),
            "spectral_bandwidth_min": float(np.min(bandwidth)),
            "spectral_bandwidth_max": float(np.max(bandwidth)),
        }
    
    def spectral_rolloff(
        self,
        audio: np.ndarray,
        roll_percent: float = 0.85
    ) -> Dict[str, float]:
        """
        Compute spectral rolloff frequency.
        
        The frequency below which a specified percentage
        of the total spectral energy lies.
        
        Args:
            audio: Input audio signal.
            roll_percent: Fraction of energy (default 85%).
            
        Returns:
            Dictionary with rolloff statistics.
        """
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            roll_percent=roll_percent
        )[0]
        
        return {
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_rolloff_std": float(np.std(rolloff)),
            "spectral_rolloff_min": float(np.min(rolloff)),
            "spectral_rolloff_max": float(np.max(rolloff)),
        }
    
    def spectral_contrast(
        self,
        audio: np.ndarray,
        n_bands: int = 6
    ) -> Dict[str, float]:
        """
        Compute spectral contrast.
        
        Measures the difference between peaks and valleys
        in the spectrum across frequency bands.
        
        Args:
            audio: Input audio signal.
            n_bands: Number of frequency bands.
            
        Returns:
            Dictionary with contrast per band and overall stats.
        """
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_bands=n_bands
        )
        
        result = {}
        for i in range(contrast.shape[0]):
            result[f"spectral_contrast_band{i}_mean"] = float(np.mean(contrast[i]))
        
        result["spectral_contrast_overall_mean"] = float(np.mean(contrast))
        result["spectral_contrast_overall_std"] = float(np.std(contrast))
        
        return result
    
    def spectral_flatness(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral flatness (tonality coefficient).
        
        Measures how noise-like (flat) vs tone-like (peaked)
        the spectrum is. Values near 1 = noise-like.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with flatness statistics.
        """
        flatness = librosa.feature.spectral_flatness(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        return {
            "spectral_flatness_mean": float(np.mean(flatness)),
            "spectral_flatness_std": float(np.std(flatness)),
            "spectral_flatness_min": float(np.min(flatness)),
            "spectral_flatness_max": float(np.max(flatness)),
        }
    
    def zero_crossing_rate(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute zero-crossing rate.
        
        Rate at which the signal changes sign. Higher values
        indicate more high-frequency content or noise.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with ZCR statistics.
        """
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        return {
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "zero_crossing_rate_std": float(np.std(zcr)),
            "zero_crossing_rate_min": float(np.min(zcr)),
            "zero_crossing_rate_max": float(np.max(zcr)),
        }
    
    def chroma_features(
        self,
        audio: np.ndarray,
        n_chroma: int = 12
    ) -> Dict[str, float]:
        """
        Compute chroma (pitch class) features.
        
        Projects the spectrum onto 12 pitch classes,
        useful for harmonic analysis.
        
        Args:
            audio: Input audio signal.
            n_chroma: Number of chroma bins.
            
        Returns:
            Dictionary with chroma statistics.
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=n_chroma
        )
        
        result = {}
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                       'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i, name in enumerate(pitch_names):
            result[f"chroma_{name}_mean"] = float(np.mean(chroma[i]))
        
        # Overall chroma statistics
        result["chroma_mean"] = float(np.mean(chroma))
        result["chroma_std"] = float(np.std(chroma))
        
        return result
    
    def extract_all(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract all spectral features.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary containing all spectral features.
        """
        features = {}
        
        features.update(self.spectral_centroid(audio))
        features.update(self.spectral_bandwidth(audio))
        features.update(self.spectral_rolloff(audio))
        features.update(self.spectral_contrast(audio))
        features.update(self.spectral_flatness(audio))
        features.update(self.zero_crossing_rate(audio))
        features.update(self.chroma_features(audio))
        
        return features
