"""
Audio synthesis and reconstruction.

Reconstructs audio from extracted features using:
- Griffin-Lim algorithm (from mel spectrograms)
- Vocoder-based synthesis (from pitch + spectral envelope)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import librosa


class AudioSynthesizer:
    """
    Synthesize audio from extracted features.
    
    Supports reconstruction from:
    - Mel spectrograms (Griffin-Lim algorithm)
    - STFT magnitude (Griffin-Lim)
    - Mel + Pitch (vocoder-style)
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_iter: int = 32
    ):
        """
        Initialize audio synthesizer.
        
        Args:
            sample_rate: Target sample rate.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            n_mels: Number of mel bands.
            n_iter: Griffin-Lim iterations (more = better quality).
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_iter = n_iter
        
        # Create mel filterbank for inversion
        self._mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
    
    def extract_for_synthesis(
        self,
        audio: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract features needed for synthesis.
        
        Stores all information needed to reconstruct the audio.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Dictionary with synthesis features:
                - mel_spectrogram: Mel spectrogram (power)
                - stft_magnitude: STFT magnitude
                - stft_phase: STFT phase (for perfect reconstruction)
                - pitch: F0 contour
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Extract pitch
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=100,
            fmax=8000,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return {
            "mel_spectrogram": mel_spec,
            "stft_magnitude": magnitude,
            "stft_phase": phase,
            "pitch": f0,
            "voiced_flag": voiced_flag,
            "sample_rate": self.sample_rate,
        }
    
    def synthesize_from_mel(
        self,
        mel_spectrogram: np.ndarray
    ) -> np.ndarray:
        """
        Synthesize audio from mel spectrogram using Griffin-Lim.
        
        This is an iterative algorithm that estimates phase
        from magnitude, producing reasonable reconstructions.
        
        Args:
            mel_spectrogram: Mel spectrogram (power).
            
        Returns:
            Reconstructed audio signal.
        """
        # Invert mel spectrogram to linear spectrogram
        # Using pseudo-inverse of mel filterbank
        mel_basis_pinv = np.linalg.pinv(self._mel_basis)
        magnitude = np.maximum(0, np.dot(mel_basis_pinv, mel_spectrogram))
        
        # Apply Griffin-Lim to estimate phase
        audio = librosa.griffinlim(
            magnitude,
            n_iter=self.n_iter,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        return audio
    
    def synthesize_from_stft(
        self,
        magnitude: np.ndarray,
        phase: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesize audio from STFT magnitude.
        
        If phase is provided, uses exact reconstruction.
        Otherwise uses Griffin-Lim to estimate phase.
        
        Args:
            magnitude: STFT magnitude.
            phase: Optional STFT phase.
            
        Returns:
            Reconstructed audio signal.
        """
        if phase is not None:
            # Perfect reconstruction with original phase
            stft = magnitude * np.exp(1j * phase)
            audio = librosa.istft(stft, hop_length=self.hop_length)
        else:
            # Griffin-Lim phase estimation
            audio = librosa.griffinlim(
                magnitude,
                n_iter=self.n_iter,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
        
        return audio
    
    def synthesize_from_features(
        self,
        features: Dict[str, np.ndarray],
        use_original_phase: bool = False
    ) -> np.ndarray:
        """
        Synthesize audio from extracted features.
        
        Args:
            features: Dictionary from extract_for_synthesis().
            use_original_phase: If True, use stored phase for
                               perfect reconstruction.
            
        Returns:
            Reconstructed audio signal.
        """
        if use_original_phase and "stft_phase" in features:
            return self.synthesize_from_stft(
                features["stft_magnitude"],
                features["stft_phase"]
            )
        elif "mel_spectrogram" in features:
            return self.synthesize_from_mel(features["mel_spectrogram"])
        elif "stft_magnitude" in features:
            return self.synthesize_from_stft(features["stft_magnitude"])
        else:
            raise ValueError("No suitable features for synthesis")
    
    def synthesize_with_modifications(
        self,
        features: Dict[str, np.ndarray],
        pitch_shift: float = 0.0,
        time_stretch: float = 1.0,
        spectral_shift: float = 0.0
    ) -> np.ndarray:
        """
        Synthesize audio with modifications.
        
        Args:
            features: Synthesis features dictionary.
            pitch_shift: Semitones to shift pitch (positive = higher).
            time_stretch: Time stretch factor (>1 = slower).
            spectral_shift: Shift spectrum up/down.
            
        Returns:
            Modified synthesized audio.
        """
        mel_spec = features["mel_spectrogram"].copy()
        
        # Apply spectral shift (shift mel bands)
        if spectral_shift != 0:
            shift_bins = int(spectral_shift * self.n_mels / 10)
            if shift_bins > 0:
                mel_spec = np.roll(mel_spec, shift_bins, axis=0)
                mel_spec[:shift_bins] = 0
            elif shift_bins < 0:
                mel_spec = np.roll(mel_spec, shift_bins, axis=0)
                mel_spec[shift_bins:] = 0
        
        # Synthesize
        audio = self.synthesize_from_mel(mel_spec)
        
        # Apply pitch shift
        if pitch_shift != 0:
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=pitch_shift
            )
        
        # Apply time stretch
        if time_stretch != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=1/time_stretch)
        
        return audio


class SynthesisComparator:
    """
    Compare original and synthesized audio.
    
    Provides metrics and visualizations for comparing
    the quality of audio reconstruction.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize comparator.
        
        Args:
            sample_rate: Audio sample rate.
        """
        self.sample_rate = sample_rate
    
    def compute_comparison_metrics(
        self,
        original: np.ndarray,
        synthesized: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute metrics comparing original and synthesized audio.
        
        Args:
            original: Original audio signal.
            synthesized: Synthesized audio signal.
            
        Returns:
            Dictionary of comparison metrics.
        """
        # Ensure same length
        min_len = min(len(original), len(synthesized))
        original = original[:min_len]
        synthesized = synthesized[:min_len]
        
        # Mean squared error
        mse = np.mean((original - synthesized) ** 2)
        
        # Signal-to-noise ratio
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - synthesized) ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Correlation
        correlation = np.corrcoef(original, synthesized)[0, 1]
        
        # Spectral similarity
        orig_spec = np.abs(librosa.stft(original))
        synth_spec = np.abs(librosa.stft(synthesized))
        
        # Ensure same shape
        min_frames = min(orig_spec.shape[1], synth_spec.shape[1])
        orig_spec = orig_spec[:, :min_frames]
        synth_spec = synth_spec[:, :min_frames]
        
        spectral_correlation = np.corrcoef(
            orig_spec.flatten(),
            synth_spec.flatten()
        )[0, 1]
        
        # Mel spectrogram similarity
        orig_mel = librosa.feature.melspectrogram(y=original, sr=self.sample_rate)
        synth_mel = librosa.feature.melspectrogram(y=synthesized, sr=self.sample_rate)
        
        min_frames = min(orig_mel.shape[1], synth_mel.shape[1])
        orig_mel = orig_mel[:, :min_frames]
        synth_mel = synth_mel[:, :min_frames]
        
        mel_correlation = np.corrcoef(
            orig_mel.flatten(),
            synth_mel.flatten()
        )[0, 1]
        
        return {
            "mse": float(mse),
            "snr_db": float(snr),
            "waveform_correlation": float(correlation),
            "spectral_correlation": float(spectral_correlation),
            "mel_correlation": float(mel_correlation),
        }
    
    def extract_comparison_features(
        self,
        original: np.ndarray,
        synthesized: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from both for visual comparison.
        
        Args:
            original: Original audio.
            synthesized: Synthesized audio.
            n_fft: FFT size.
            hop_length: Hop length.
            
        Returns:
            Dictionary with spectrograms for both signals.
        """
        # Ensure same length
        min_len = min(len(original), len(synthesized))
        original = original[:min_len]
        synthesized = synthesized[:min_len]
        
        # Compute spectrograms
        orig_spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(original, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        synth_spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(synthesized, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        
        # Compute mel spectrograms
        orig_mel = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=original, sr=self.sample_rate,
                n_fft=n_fft, hop_length=hop_length
            ),
            ref=np.max
        )
        synth_mel = librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=synthesized, sr=self.sample_rate,
                n_fft=n_fft, hop_length=hop_length
            ),
            ref=np.max
        )
        
        # Difference spectrogram
        min_frames = min(orig_spec.shape[1], synth_spec.shape[1])
        difference = orig_spec[:, :min_frames] - synth_spec[:, :min_frames]
        
        return {
            "original_spectrogram": orig_spec,
            "synthesized_spectrogram": synth_spec,
            "original_mel": orig_mel,
            "synthesized_mel": synth_mel,
            "difference": difference,
            "original_waveform": original,
            "synthesized_waveform": synthesized,
        }
