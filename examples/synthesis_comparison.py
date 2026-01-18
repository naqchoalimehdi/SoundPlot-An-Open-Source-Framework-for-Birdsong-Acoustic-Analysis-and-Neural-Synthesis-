"""
Synthesis comparison example.

Demonstrates the full round-trip:
1. Load audio
2. Extract features for synthesis
3. Synthesize audio from features
4. Compare original vs synthesized
5. Visualize the comparison
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.audio import AudioLoader, AudioPreprocessor, AudioSynthesizer, SynthesisComparator
from src.features import FeatureExtractor
from src.analysis import DimensionalityReducer
from src.utils import ComparisonVisualizer
import soundfile as sf


def synthesize_and_compare(
    audio_path: str,
    output_dir: str = "data/comparisons"
):
    """
    Run full synthesis and comparison pipeline.
    
    Args:
        audio_path: Path to audio file.
        output_dir: Directory for output files.
    """
    print(f"Processing: {audio_path}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load audio
    print("\n1. Loading audio...")
    loader = AudioLoader(target_sr=22050)
    audio, sr = loader.load(audio_path)
    print(f"   Loaded {len(audio)/sr:.2f} seconds")
    
    # 2. Preprocess
    print("\n2. Preprocessing...")
    preprocessor = AudioPreprocessor(sample_rate=sr)
    audio = preprocessor.normalize(audio)
    
    # 3. Extract synthesis features
    print("\n3. Extracting synthesis features...")
    synthesizer = AudioSynthesizer(sample_rate=sr)
    synth_features = synthesizer.extract_for_synthesis(audio)
    print(f"   Mel shape: {synth_features['mel_spectrogram'].shape}")
    
    # 4. Synthesize from mel spectrogram
    print("\n4. Synthesizing audio from mel spectrogram...")
    synthesized = synthesizer.synthesize_from_mel(synth_features["mel_spectrogram"])
    synthesized = preprocessor.normalize(synthesized)
    print(f"   Synthesized {len(synthesized)/sr:.2f} seconds")
    
    # 5. Save synthesized audio
    synth_audio_path = output_path / "synthesized.wav"
    sf.write(str(synth_audio_path), synthesized, sr)
    print(f"   Saved to: {synth_audio_path}")
    
    # 6. Compare
    print("\n5. Computing comparison metrics...")
    comparator = SynthesisComparator(sample_rate=sr)
    metrics = comparator.compute_comparison_metrics(audio, synthesized)
    print(f"   SNR: {metrics['snr_db']:.2f} dB")
    print(f"   Spectral correlation: {metrics['spectral_correlation']:.3f}")
    print(f"   Mel correlation: {metrics['mel_correlation']:.3f}")
    
    # 7. Extract comparison features
    print("\n6. Extracting comparison features...")
    comparison_features = comparator.extract_comparison_features(audio, synthesized)
    
    # 8. Visualize
    print("\n7. Creating comparison visualization...")
    try:
        visualizer = ComparisonVisualizer(output_dir=str(output_path))
        plot_path = visualizer.plot_full_comparison(
            comparison_features,
            metrics,
            sample_rate=sr,
            title="Original vs Synthesized Birdsong",
            filename="comparison"
        )
        print(f"   Saved visualization: {plot_path}")
    except ImportError:
        print("   Matplotlib not available, skipping visualization")
    
    # 9. Compare embeddings
    print("\n8. Comparing feature embeddings...")
    extractor = FeatureExtractor(sample_rate=sr)
    
    # Segment both and extract features
    orig_segments = preprocessor.segment(audio, 0.2, 0.1)
    synth_segments = preprocessor.segment(synthesized, 0.2, 0.1)
    
    # Use minimum of both lengths
    n_segments = min(len(orig_segments), len(synth_segments))
    
    orig_features = []
    synth_features_list = []
    
    for i in range(n_segments):
        orig_feat = list(extractor.extract_compact(orig_segments[i][0]).values())
        synth_feat = list(extractor.extract_compact(synth_segments[i][0]).values())
        orig_features.append(orig_feat)
        synth_features_list.append(synth_feat)
    
    orig_features = np.array(orig_features)
    synth_features_arr = np.array(synth_features_list)
    
    # Reduce dimensions
    reducer = DimensionalityReducer(n_components=2, method="pca")
    
    # Fit on original, transform both
    reducer.fit(orig_features)
    orig_embeddings = reducer.transform(orig_features)
    synth_embeddings = reducer.transform(synth_features_arr)
    
    # Compute embedding drift
    drift = np.mean(np.linalg.norm(orig_embeddings - synth_embeddings, axis=1))
    print(f"   Mean embedding drift: {drift:.3f}")
    
    # Visualize embedding comparison
    try:
        plot_path = visualizer.plot_embedding_comparison(
            orig_embeddings,
            synth_embeddings,
            title="Feature Space: Original vs Synthesized",
            filename="embedding_comparison"
        )
        print(f"   Saved embedding comparison: {plot_path}")
    except Exception:
        pass
    
    print("\n✓ Synthesis and comparison complete!")
    
    return {
        "metrics": metrics,
        "synthesized_audio": synthesized,
        "embedding_drift": drift,
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Synthesize and compare birdsong audio"
    )
    parser.add_argument(
        "audio_path",
        help="Path to audio file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/comparisons",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    synthesize_and_compare(args.audio_path, args.output_dir)


if __name__ == "__main__":
    main()
