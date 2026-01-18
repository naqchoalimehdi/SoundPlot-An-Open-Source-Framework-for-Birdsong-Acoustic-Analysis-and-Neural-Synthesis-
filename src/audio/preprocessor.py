"""
Audio preprocessing for birdsong analysis.

Provides noise reduction, normalization, silence removal,
and segmentation utilities to prepare audio for feature extraction.
"""

from typing import Tuple, List, Optional
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d


class AudioPreprocessor:
    """
    Preprocess audio signals for feature extraction.
    
    Handles common preprocessing steps including:
    - Amplitude normalization
    - Noise reduction (spectral gating)
    - Silence removal
    - Segmentation into analysis windows
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize preprocessor.
        
        Args:
            sample_rate: Sample rate of audio to process.
        """
        self.sample_rate = sample_rate
    
    def normalize(
        self,
        audio: np.ndarray,
        method: str = "peak"
    ) -> np.ndarray:
        """
        Normalize audio amplitude.
        
        Args:
            audio: Input audio signal.
            method: Normalization method:
                - "peak": Scale to [-1, 1] based on peak amplitude
                - "rms": Scale based on RMS energy
                - "lufs": Loudness normalization (approximate)
                
        Returns:
            Normalized audio signal.
        """
        if len(audio) == 0:
            return audio
        
        if method == "peak":
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio / peak
            return audio
        
        elif method == "rms":
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                target_rms = 0.1  # Target RMS level
                return audio * (target_rms / rms)
            return audio
        
        elif method == "lufs":
            # Approximate loudness normalization
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                target_lufs = -23  # EBU R128 standard
                current_lufs = 20 * np.log10(rms) if rms > 0 else -100
                gain_db = target_lufs - current_lufs
                gain_linear = 10 ** (gain_db / 20)
                normalized = audio * gain_linear
                # Prevent clipping
                peak = np.max(np.abs(normalized))
                if peak > 1.0:
                    normalized = normalized / peak
                return normalized
            return audio
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def reduce_noise(
        self,
        audio: np.ndarray,
        noise_reduce_strength: float = 0.5,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Reduce background noise using spectral gating.
        
        This method estimates a noise floor from quieter parts
        of the signal and subtracts it from the spectrum.
        
        Args:
            audio: Input audio signal.
            noise_reduce_strength: How aggressively to reduce noise (0-1).
            n_fft: FFT size.
            hop_length: Hop length for STFT.
            
        Returns:
            Noise-reduced audio signal.
        """
        if len(audio) < n_fft:
            return audio
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor (use lower percentile of magnitude)
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Spectral subtraction with flooring
        magnitude_denoised = magnitude - (noise_reduce_strength * noise_floor)
        magnitude_denoised = np.maximum(magnitude_denoised, magnitude * 0.1)
        
        # Reconstruct signal
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length)
        
        return audio_denoised
    
    def remove_silence(
        self,
        audio: np.ndarray,
        top_db: float = 30,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Remove silent portions from audio.
        
        Args:
            audio: Input audio signal.
            top_db: Threshold in dB below reference to consider silence.
            frame_length: Frame length for energy computation.
            hop_length: Hop length for analysis.
            
        Returns:
            Tuple of:
                - Audio with silence removed
                - List of (start, end) sample indices of non-silent regions
        """
        # Get non-silent intervals
        intervals = librosa.effects.split(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        if len(intervals) == 0:
            return audio, [(0, len(audio))]
        
        # Concatenate non-silent segments
        non_silent = np.concatenate([
            audio[start:end] for start, end in intervals
        ])
        
        return non_silent, [(int(s), int(e)) for s, e in intervals]
    
    def segment(
        self,
        audio: np.ndarray,
        segment_duration: float = 1.0,
        hop_duration: float = 0.5,
        min_duration: float = 0.1
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Segment audio into overlapping windows.
        
        Args:
            audio: Input audio signal.
            segment_duration: Duration of each segment in seconds.
            hop_duration: Hop between segments in seconds.
            min_duration: Minimum segment duration to include.
            
        Returns:
            List of (segment_audio, start_time, end_time) tuples.
        """
        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(hop_duration * self.sample_rate)
        min_samples = int(min_duration * self.sample_rate)
        
        segments = []
        start = 0
        
        while start < len(audio):
            end = min(start + segment_samples, len(audio))
            segment = audio[start:end]
            
            if len(segment) >= min_samples:
                start_time = start / self.sample_rate
                end_time = end / self.sample_rate
                segments.append((segment, start_time, end_time))
            
            start += hop_samples
        
        return segments
    
    def detect_calls(
        self,
        audio: np.ndarray,
        threshold_db: float = -30,
        min_duration: float = 0.05,
        max_duration: float = 5.0,
        merge_gap: float = 0.1
    ) -> List[Tuple[float, float]]:
        """
        Detect individual bird calls in audio.
        
        Uses onset detection combined with energy thresholding
        to identify call boundaries.
        
        Args:
            audio: Input audio signal.
            threshold_db: Energy threshold in dB.
            min_duration: Minimum call duration in seconds.
            max_duration: Maximum call duration in seconds.
            merge_gap: Gap in seconds below which to merge segments.
            
        Returns:
            List of (start_time, end_time) tuples for each detected call.
        """
        # Compute frame-wise RMS energy
        hop_length = 512
        frame_length = 2048
        
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find frames above threshold
        active_frames = rms_db > threshold_db
        
        # Convert to sample indices
        frame_times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # Find continuous regions
        calls = []
        in_call = False
        call_start = 0
        
        for i, active in enumerate(active_frames):
            if active and not in_call:
                in_call = True
                call_start = frame_times[i]
            elif not active and in_call:
                in_call = False
                call_end = frame_times[i]
                duration = call_end - call_start
                if min_duration <= duration <= max_duration:
                    calls.append((call_start, call_end))
        
        # Handle call extending to end
        if in_call:
            call_end = frame_times[-1]
            duration = call_end - call_start
            if min_duration <= duration <= max_duration:
                calls.append((call_start, call_end))
        
        # Merge close calls
        if len(calls) > 1 and merge_gap > 0:
            merged = [calls[0]]
            for start, end in calls[1:]:
                prev_end = merged[-1][1]
                if start - prev_end <= merge_gap:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
            calls = merged
        
        return calls
    
    def preprocess_full(
        self,
        audio: np.ndarray,
        normalize: bool = True,
        denoise: bool = True,
        remove_silence: bool = False
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            audio: Input audio signal.
            normalize: Whether to normalize amplitude.
            denoise: Whether to apply noise reduction.
            remove_silence: Whether to remove silent portions.
            
        Returns:
            Preprocessed audio signal.
        """
        result = audio.copy()
        
        if denoise:
            result = self.reduce_noise(result)
        
        if remove_silence:
            result, _ = self.remove_silence(result)
        
        if normalize:
            result = self.normalize(result)
        
        return result
