"""
MFCC (Mel-Frequency Cepstral Coefficients) feature extraction.

MFCCs are the most widely used features for audio analysis,
capturing timbral characteristics in a perceptually-relevant way.
"""

from typing import Dict, Optional
import numpy as np
import librosa


class MFCCFeatures:
    """
    Extract MFCC features from audio signals.
    
    MFCCs represent the short-term power spectrum of sound
    on a mel scale, mimicking human auditory perception.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """
        Initialize MFCC feature extractor.
        
        Args:
            sample_rate: Audio sample rate.
            n_mfcc: Number of MFCCs to extract.
            n_fft: FFT window size.
            hop_length: Number of samples between frames.
            n_mels: Number of mel filterbank channels.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract raw MFCC matrix.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            MFCC matrix of shape (n_mfcc, n_frames).
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        return mfcc
    
    def extract_delta(
        self,
        audio: np.ndarray,
        order: int = 1
    ) -> np.ndarray:
        """
        Extract MFCC delta (derivative) features.
        
        Delta features capture the dynamics/change of MFCCs.
        
        Args:
            audio: Input audio signal.
            order: Order of derivative (1=delta, 2=delta-delta).
            
        Returns:
            Delta MFCC matrix.
        """
        mfcc = self.extract_mfcc(audio)
        
        delta = librosa.feature.delta(mfcc, order=order)
        
        return delta
    
    def mfcc_statistics(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical summaries of MFCCs.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with per-coefficient statistics.
        """
        mfcc = self.extract_mfcc(audio)
        
        features = {}
        
        for i in range(self.n_mfcc):
            coef = mfcc[i]
            features[f"mfcc_{i}_mean"] = float(np.mean(coef))
            features[f"mfcc_{i}_std"] = float(np.std(coef))
            features[f"mfcc_{i}_min"] = float(np.min(coef))
            features[f"mfcc_{i}_max"] = float(np.max(coef))
        
        return features
    
    def delta_statistics(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical summaries of delta MFCCs.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with per-coefficient delta statistics.
        """
        delta = self.extract_delta(audio, order=1)
        delta2 = self.extract_delta(audio, order=2)
        
        features = {}
        
        for i in range(self.n_mfcc):
            # First derivative (velocity)
            features[f"mfcc_delta_{i}_mean"] = float(np.mean(delta[i]))
            features[f"mfcc_delta_{i}_std"] = float(np.std(delta[i]))
            
            # Second derivative (acceleration)
            features[f"mfcc_delta2_{i}_mean"] = float(np.mean(delta2[i]))
            features[f"mfcc_delta2_{i}_std"] = float(np.std(delta2[i]))
        
        return features
    
    def mfcc_vector(
        self,
        audio: np.ndarray,
        include_delta: bool = True,
        include_delta2: bool = True
    ) -> np.ndarray:
        """
        Get a fixed-length MFCC feature vector.
        
        Computes mean and std of MFCCs (and optionally deltas)
        to create a single feature vector per audio segment.
        
        Args:
            audio: Input audio signal.
            include_delta: Include first derivative.
            include_delta2: Include second derivative.
            
        Returns:
            1D feature vector.
        """
        mfcc = self.extract_mfcc(audio)
        
        vectors = [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1)
        ]
        
        if include_delta:
            delta = self.extract_delta(audio, order=1)
            vectors.extend([
                np.mean(delta, axis=1),
                np.std(delta, axis=1)
            ])
        
        if include_delta2:
            delta2 = self.extract_delta(audio, order=2)
            vectors.extend([
                np.mean(delta2, axis=1),
                np.std(delta2, axis=1)
            ])
        
        return np.concatenate(vectors)
    
    def mel_spectrogram_stats(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute mel spectrogram statistics.
        
        These complement MFCCs with additional spectral information.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with mel spectrogram statistics.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            "mel_spec_mean": float(np.mean(mel_db)),
            "mel_spec_std": float(np.std(mel_db)),
            "mel_spec_max": float(np.max(mel_db)),
            "mel_spec_min": float(np.min(mel_db)),
        }
    
    def extract_all(
        self,
        audio: np.ndarray,
        include_delta: bool = True
    ) -> Dict[str, float]:
        """
        Extract all MFCC-based features.
        
        Args:
            audio: Input audio signal.
            include_delta: Whether to include delta features.
            
        Returns:
            Dictionary containing all MFCC features.
        """
        features = {}
        
        features.update(self.mfcc_statistics(audio))
        
        if include_delta:
            features.update(self.delta_statistics(audio))
        
        features.update(self.mel_spectrogram_stats(audio))
        
        return features
