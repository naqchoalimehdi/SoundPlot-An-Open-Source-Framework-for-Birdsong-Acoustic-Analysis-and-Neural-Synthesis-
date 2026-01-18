"""
Temporal feature extraction for audio analysis.

Extracts time-domain features including onset detection,
rhythm analysis, tempo estimation, and energy dynamics.
"""

from typing import Dict, List, Tuple
import numpy as np
import librosa


class TemporalFeatures:
    """
    Extract temporal/rhythmic features from audio signals.
    
    Features include onset detection, tempo, beat tracking,
    and inter-onset interval analysis.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512
    ):
        """
        Initialize temporal feature extractor.
        
        Args:
            sample_rate: Audio sample rate.
            hop_length: Number of samples between frames.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def onset_detection(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Detect note/call onsets and compute statistics.
        
        Onsets mark the beginning of acoustic events.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with onset count and rate.
        """
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Detect onset frames
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Convert to times
        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        duration = len(audio) / self.sample_rate
        
        return {
            "onset_count": len(onset_times),
            "onset_rate": len(onset_times) / duration if duration > 0 else 0,
            "onset_strength_mean": float(np.mean(onset_env)),
            "onset_strength_std": float(np.std(onset_env)),
            "onset_strength_max": float(np.max(onset_env)),
        }
    
    def get_onset_times(self, audio: np.ndarray) -> np.ndarray:
        """
        Get onset times in seconds.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Array of onset times.
        """
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return librosa.frames_to_time(
            onset_frames,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
    
    def inter_onset_intervals(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute inter-onset interval (IOI) statistics.
        
        IOIs capture the timing patterns between acoustic events.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with IOI statistics.
        """
        onset_times = self.get_onset_times(audio)
        
        if len(onset_times) < 2:
            return {
                "ioi_mean": 0.0,
                "ioi_std": 0.0,
                "ioi_min": 0.0,
                "ioi_max": 0.0,
                "ioi_cv": 0.0,  # Coefficient of variation
            }
        
        iois = np.diff(onset_times)
        
        return {
            "ioi_mean": float(np.mean(iois)),
            "ioi_std": float(np.std(iois)),
            "ioi_min": float(np.min(iois)),
            "ioi_max": float(np.max(iois)),
            "ioi_cv": float(np.std(iois) / np.mean(iois)) if np.mean(iois) > 0 else 0,
        }
    
    def tempo_estimation(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Estimate tempo/pace of the audio.
        
        Tempo estimation is based on beat tracking,
        adapted for birdsong rhythm analysis.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with tempo in BPM and confidence.
        """
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Estimate tempo
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Handle both old and new librosa API
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo = float(tempo)
        
        return {
            "tempo_bpm": tempo,
            "beat_count": len(beat_frames),
            "beat_regularity": self._compute_beat_regularity(beat_frames),
        }
    
    def _compute_beat_regularity(self, beat_frames: np.ndarray) -> float:
        """Compute how regular the beat pattern is."""
        if len(beat_frames) < 3:
            return 0.0
        
        intervals = np.diff(beat_frames)
        if len(intervals) == 0 or np.mean(intervals) == 0:
            return 0.0
        
        # Coefficient of variation (lower = more regular)
        cv = np.std(intervals) / np.mean(intervals)
        # Convert to regularity score (higher = more regular)
        regularity = 1.0 / (1.0 + cv)
        
        return float(regularity)
    
    def energy_dynamics(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute energy/loudness dynamics over time.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with RMS energy statistics.
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=self.hop_length
        )[0]
        
        # Compute dynamics
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        return {
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_min": float(np.min(rms)),
            "rms_max": float(np.max(rms)),
            "dynamic_range_db": float(np.max(rms_db) - np.min(rms_db)),
            "rms_db_mean": float(np.mean(rms_db)),
        }
    
    def duration_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute duration-related features.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with duration features.
        """
        duration = len(audio) / self.sample_rate
        
        # Compute effective duration (excluding very quiet parts)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=self.hop_length
        )[0]
        
        threshold = np.max(rms) * 0.1  # 10% of max
        active_frames = np.sum(rms > threshold)
        effective_duration = active_frames * self.hop_length / self.sample_rate
        
        return {
            "duration_seconds": duration,
            "effective_duration_seconds": float(effective_duration),
            "duty_cycle": float(effective_duration / duration) if duration > 0 else 0,
        }
    
    def extract_all(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract all temporal features.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary containing all temporal features.
        """
        features = {}
        
        features.update(self.onset_detection(audio))
        features.update(self.inter_onset_intervals(audio))
        features.update(self.tempo_estimation(audio))
        features.update(self.energy_dynamics(audio))
        features.update(self.duration_features(audio))
        
        return features
