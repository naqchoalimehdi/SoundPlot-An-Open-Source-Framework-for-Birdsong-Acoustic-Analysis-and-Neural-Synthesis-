"""
Generate synthetic birdsong for testing.

Creates realistic-sounding bird call audio with:
- Frequency sweeps (chirps)
- Multiple harmonics
- Rhythmic patterns
- Natural variations
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_chirp(sr: int, duration: float, f_start: float, f_end: float) -> np.ndarray:
    """Generate a frequency sweep (chirp)."""
    t = np.linspace(0, duration, int(sr * duration))
    # Exponential frequency sweep
    freq = f_start * (f_end / f_start) ** (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    # Apply envelope
    envelope = np.sin(np.pi * t / duration) ** 2
    return envelope * np.sin(phase)


def generate_trill(sr: int, duration: float, base_freq: float, trill_rate: float) -> np.ndarray:
    """Generate a trilling sound."""
    t = np.linspace(0, duration, int(sr * duration))
    # Frequency modulation
    freq_mod = base_freq + 200 * np.sin(2 * np.pi * trill_rate * t)
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    # Envelope with slight decay
    envelope = np.exp(-t / duration) * np.sin(np.pi * t / duration)
    return envelope * np.sin(phase)


def generate_whistle(sr: int, duration: float, freq: float) -> np.ndarray:
    """Generate a pure whistle tone with harmonics."""
    t = np.linspace(0, duration, int(sr * duration))
    # Add harmonics
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * freq * t)  # 2nd harmonic
    signal += 0.25 * np.sin(2 * np.pi * 3 * freq * t)  # 3rd harmonic
    # Envelope
    envelope = np.sin(np.pi * t / duration) ** 0.5
    return envelope * signal


def add_noise(signal: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    """Add subtle background noise."""
    noise = np.random.randn(len(signal)) * noise_level
    return signal + noise


def generate_birdsong(duration: float = 10.0, sr: int = 22050) -> np.ndarray:
    """
    Generate a synthetic birdsong sequence.
    
    Creates a sequence of different call types with natural timing.
    """
    total_samples = int(duration * sr)
    audio = np.zeros(total_samples)
    
    # Different call patterns
    call_types = [
        ("chirp_up", lambda: generate_chirp(sr, 0.15, 2000, 4000)),
        ("chirp_down", lambda: generate_chirp(sr, 0.12, 4500, 2500)),
        ("trill", lambda: generate_trill(sr, 0.3, 3000, 25)),
        ("whistle_high", lambda: generate_whistle(sr, 0.2, 4000)),
        ("whistle_low", lambda: generate_whistle(sr, 0.25, 2500)),
        ("double_chirp", lambda: np.concatenate([
            generate_chirp(sr, 0.08, 2500, 3500),
            np.zeros(int(sr * 0.03)),
            generate_chirp(sr, 0.08, 2500, 3500)
        ])),
    ]
    
    # Generate sequence with natural timing
    position = int(sr * 0.5)  # Start after 0.5s
    
    np.random.seed(42)  # Reproducible
    
    while position < total_samples - sr:
        # Pick random call type
        call_name, call_func = call_types[np.random.randint(len(call_types))]
        call = call_func()
        
        # Random amplitude variation
        amplitude = 0.3 + 0.4 * np.random.random()
        call = call * amplitude
        
        # Add to audio
        end_pos = min(position + len(call), total_samples)
        audio[position:end_pos] = call[:end_pos - position]
        
        # Gap between calls (100-500ms)
        gap = int(sr * (0.1 + 0.4 * np.random.random()))
        position += len(call) + gap
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Add subtle ambient noise
    audio = add_noise(audio, 0.01)
    
    return audio


def main():
    """Generate and save synthetic birdsong samples."""
    
    output_dir = Path(__file__).parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic birdsong audio...")
    
    sr = 22050
    
    # Generate multiple samples with different characteristics
    samples = [
        ("birdsong_fast.wav", 8.0, 42),
        ("birdsong_slow.wav", 10.0, 123),
        ("birdsong_varied.wav", 12.0, 456),
    ]
    
    generated = []
    
    for filename, duration, seed in samples:
        print(f"\n  Generating {filename}...")
        np.random.seed(seed)
        audio = generate_birdsong(duration=duration, sr=sr)
        
        output_path = output_dir / filename
        sf.write(str(output_path), audio, sr)
        
        print(f"    Duration: {duration}s")
        print(f"    Saved to: {output_path}")
        generated.append(output_path)
    
    print(f"\n✓ Generated {len(generated)} birdsong samples!")
    print(f"  Location: {output_dir}")
    
    return generated


if __name__ == "__main__":
    files = main()
    
    if files:
        print("\n" + "="*50)
        print("To run the full analysis pipeline:")
        print(f'  python examples/basic_analysis.py "{files[0]}"')
        print("\nTo run synthesis comparison:")
        print(f'  python examples/synthesis_comparison.py "{files[0]}"')
