"""
Audio file loading and validation.

Handles loading of various audio formats (WAV, MP3, FLAC, OGG)
with automatic mono conversion and sample rate standardization.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import librosa
import soundfile as sf


class AudioLoader:
    """
    Load and validate audio files for analysis.
    
    Supports batch loading from directories and handles
    format conversion, stereo-to-mono mixing, and resampling.
    
    Attributes:
        target_sr: Target sample rate for all loaded audio.
        mono: Whether to convert stereo to mono.
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    def __init__(
        self,
        target_sr: int = 22050,
        mono: bool = True
    ):
        """
        Initialize the audio loader.
        
        Args:
            target_sr: Target sample rate. Default 22050 Hz is standard
                      for audio analysis (captures up to ~11 kHz).
            mono: Convert stereo to mono if True.
        """
        self.target_sr = target_sr
        self.mono = mono
    
    def load(
        self,
        file_path: Union[str, Path]
    ) -> Tuple[np.ndarray, int]:
        """
        Load a single audio file.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Tuple of (audio_signal, sample_rate):
                - audio_signal: 1D numpy array of audio samples
                - sample_rate: The sample rate of the returned audio
                
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is not supported.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        # Load with librosa (handles resampling and mono conversion)
        audio, sr = librosa.load(
            str(file_path),
            sr=self.target_sr,
            mono=self.mono
        )
        
        return audio, sr
    
    def load_segment(
        self,
        file_path: Union[str, Path],
        start_time: float,
        duration: float
    ) -> Tuple[np.ndarray, int]:
        """
        Load a segment of an audio file.
        
        Args:
            file_path: Path to the audio file.
            start_time: Start time in seconds.
            duration: Duration in seconds.
            
        Returns:
            Tuple of (audio_segment, sample_rate).
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        audio, sr = librosa.load(
            str(file_path),
            sr=self.target_sr,
            mono=self.mono,
            offset=start_time,
            duration=duration
        )
        
        return audio, sr
    
    def load_batch(
        self,
        directory: Union[str, Path],
        recursive: bool = False
    ) -> List[Tuple[str, np.ndarray, int]]:
        """
        Load all audio files from a directory.
        
        Args:
            directory: Path to directory containing audio files.
            recursive: If True, search subdirectories recursively.
            
        Returns:
            List of tuples: (filename, audio_signal, sample_rate)
            
        Note:
            Files that fail to load are skipped with a warning printed.
        """
        directory = Path(directory)
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        # Find all audio files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        audio_files = [
            f for f in directory.glob(pattern)
            if f.suffix.lower() in self.SUPPORTED_FORMATS
        ]
        
        results = []
        for file_path in sorted(audio_files):
            try:
                audio, sr = self.load(file_path)
                results.append((str(file_path), audio, sr))
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        return results
    
    def get_duration(self, file_path: Union[str, Path]) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Duration in seconds.
        """
        file_path = Path(file_path)
        return librosa.get_duration(path=str(file_path))
    
    def get_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get metadata about an audio file.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Dictionary with file information:
                - path: Full path to file
                - duration: Duration in seconds
                - sample_rate: Original sample rate
                - channels: Number of channels
                - format: File format
        """
        file_path = Path(file_path)
        
        info = sf.info(str(file_path))
        
        return {
            "path": str(file_path),
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
        }
