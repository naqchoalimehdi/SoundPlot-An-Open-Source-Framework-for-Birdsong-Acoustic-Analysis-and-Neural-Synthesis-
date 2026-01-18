"""
Pitch (F0) feature extraction for audio analysis.

Extracts fundamental frequency features using the pYIN algorithm,
which is robust for analyzing animal vocalizations.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import librosa


class PitchFeatures:
    """
    Extract pitch (fundamental frequency) features from audio.
    
    Uses the pYIN probabilistic pitch tracking algorithm,
    which handles the complex pitch contours found in birdsong.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        fmin: float = 100.0,
        fmax: float = 8000.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize pitch feature extractor.
        
        Args:
            sample_rate: Audio sample rate.
            fmin: Minimum expected frequency (Hz).
            fmax: Maximum expected frequency (Hz).
            frame_length: Analysis frame size.
            hop_length: Number of samples between frames.
        """
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def extract_pitch(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch contour from audio.
        
        Uses the pYIN algorithm for probabilistic pitch tracking.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Tuple of:
                - f0: Fundamental frequency array (Hz)
                - voiced_flag: Boolean array indicating voiced frames
                - voiced_probs: Probability of voicing per frame
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        return f0, voiced_flag, voiced_probs
    
    def pitch_statistics(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute pitch statistics from audio.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with pitch statistics (only voiced portions).
        """
        f0, voiced_flag, voiced_probs = self.extract_pitch(audio)
        
        # Filter to voiced frames only
        voiced_f0 = f0[voiced_flag]
        
        if len(voiced_f0) == 0:
            return {
                "pitch_mean_hz": 0.0,
                "pitch_std_hz": 0.0,
                "pitch_min_hz": 0.0,
                "pitch_max_hz": 0.0,
                "pitch_range_hz": 0.0,
                "pitch_median_hz": 0.0,
                "voiced_fraction": 0.0,
                "voicing_confidence_mean": float(np.mean(voiced_probs)),
            }
        
        return {
            "pitch_mean_hz": float(np.nanmean(voiced_f0)),
            "pitch_std_hz": float(np.nanstd(voiced_f0)),
            "pitch_min_hz": float(np.nanmin(voiced_f0)),
            "pitch_max_hz": float(np.nanmax(voiced_f0)),
            "pitch_range_hz": float(np.nanmax(voiced_f0) - np.nanmin(voiced_f0)),
            "pitch_median_hz": float(np.nanmedian(voiced_f0)),
            "voiced_fraction": float(np.sum(voiced_flag) / len(voiced_flag)),
            "voicing_confidence_mean": float(np.mean(voiced_probs)),
        }
    
    def pitch_contour_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze the shape/dynamics of the pitch contour.
        
        Captures how pitch changes over time, which is
        characteristic of different call types.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with pitch contour features.
        """
        f0, voiced_flag, _ = self.extract_pitch(audio)
        voiced_f0 = f0[voiced_flag]
        
        if len(voiced_f0) < 2:
            return {
                "pitch_slope": 0.0,
                "pitch_slope_abs_mean": 0.0,
                "pitch_direction_changes": 0,
                "pitch_vibrato_rate": 0.0,
            }
        
        # Compute frame-to-frame pitch changes
        pitch_diff = np.diff(voiced_f0)
        
        # Overall slope (rising vs falling)
        indices = np.arange(len(voiced_f0))
        slope, _ = np.polyfit(indices, voiced_f0, 1) if len(voiced_f0) > 1 else (0, 0)
        
        # Number of direction changes
        direction_changes = np.sum(np.diff(np.sign(pitch_diff)) != 0)
        
        # Vibrato detection (oscillation rate)
        # Simple approach: count zero-crossings in pitch derivative
        vibrato_rate = 0.0
        if len(pitch_diff) > 0:
            zero_crossings = np.sum(np.diff(np.sign(pitch_diff)) != 0)
            duration = len(voiced_f0) * self.hop_length / self.sample_rate
            vibrato_rate = zero_crossings / (2 * duration) if duration > 0 else 0
        
        return {
            "pitch_slope": float(slope),
            "pitch_slope_abs_mean": float(np.mean(np.abs(pitch_diff))),
            "pitch_direction_changes": int(direction_changes),
            "pitch_vibrato_rate": float(vibrato_rate),
        }
    
    def pitch_intervals(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze pitch intervals (musical relationships).
        
        Computes the distribution of pitch jumps,
        useful for characterizing melodic patterns.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with pitch interval features.
        """
        f0, voiced_flag, _ = self.extract_pitch(audio)
        voiced_f0 = f0[voiced_flag]
        
        if len(voiced_f0) < 2:
            return {
                "interval_mean_semitones": 0.0,
                "interval_std_semitones": 0.0,
                "interval_max_semitones": 0.0,
                "large_interval_count": 0,
            }
        
        # Convert to cents/semitones
        # 1 semitone = 100 cents = frequency ratio of 2^(1/12)
        ratio = voiced_f0[1:] / voiced_f0[:-1]
        # Avoid log of zero/negative
        valid_ratio = ratio[ratio > 0]
        
        if len(valid_ratio) == 0:
            return {
                "interval_mean_semitones": 0.0,
                "interval_std_semitones": 0.0,
                "interval_max_semitones": 0.0,
                "large_interval_count": 0,
            }
        
        semitones = 12 * np.log2(valid_ratio)
        
        return {
            "interval_mean_semitones": float(np.mean(np.abs(semitones))),
            "interval_std_semitones": float(np.std(semitones)),
            "interval_max_semitones": float(np.max(np.abs(semitones))),
            "large_interval_count": int(np.sum(np.abs(semitones) > 3)),  # > minor 3rd
        }
    
    def extract_all(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract all pitch features.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary containing all pitch features.
        """
        features = {}
        
        features.update(self.pitch_statistics(audio))
        features.update(self.pitch_contour_features(audio))
        features.update(self.pitch_intervals(audio))
        
        return features
