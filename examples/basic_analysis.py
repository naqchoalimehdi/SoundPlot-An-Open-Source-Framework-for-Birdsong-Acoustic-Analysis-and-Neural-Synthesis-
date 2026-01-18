"""
Basic analysis example.

Demonstrates the full pipeline:
1. Load audio
2. Preprocess
3. Extract features
4. Reduce dimensions
5. Cluster
6. Detect patterns
7. Export results
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio import AudioLoader, AudioPreprocessor
from src.features import FeatureExtractor
from src.analysis import DimensionalityReducer, ClusterAnalyzer, PatternDetector
from src.utils import DataExporter, Visualizer


def analyze_audio_file(audio_path: str, output_dir: str = "data/results"):
    """
    Run complete analysis on a single audio file.
    
    Args:
        audio_path: Path to audio file.
        output_dir: Directory for output files.
    """
    print(f"Analyzing: {audio_path}")
    
    # 1. Load audio
    print("\n1. Loading audio...")
    loader = AudioLoader(target_sr=22050)
    audio, sr = loader.load(audio_path)
    print(f"   Loaded {len(audio)/sr:.2f} seconds of audio")
    
    # 2. Preprocess
    print("\n2. Preprocessing...")
    preprocessor = AudioPreprocessor(sample_rate=sr)
    audio = preprocessor.preprocess_full(audio, normalize=True, denoise=True)
    
    # 3. Segment into analysis windows
    print("\n3. Segmenting...")
    segments = preprocessor.segment(audio, segment_duration=0.5, hop_duration=0.25)
    print(f"   Created {len(segments)} segments")
    
    if len(segments) < 5:
        print("   Warning: Very few segments. Try a shorter segment_duration.")
        return
    
    # 4. Extract features
    print("\n4. Extracting features...")
    extractor = FeatureExtractor(sample_rate=sr)
    
    features = []
    metadata = []
    
    for i, (seg_audio, start, end) in enumerate(segments):
        feat = extractor.extract_compact(seg_audio)
        features.append(list(feat.values()))
        metadata.append({
            "segment_id": i,
            "start_time": start,
            "end_time": end,
        })
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{len(segments)} segments")
    
    import numpy as np
    features = np.array(features)
    print(f"   Feature shape: {features.shape}")
    
    # 5. Reduce dimensions
    print("\n5. Reducing dimensions...")
    reducer = DimensionalityReducer(n_components=3, method="umap")
    
    try:
        embeddings = reducer.fit_transform(features)
        print(f"   Embedding shape: {embeddings.shape}")
    except Exception as e:
        print(f"   UMAP failed, using PCA: {e}")
        reducer = DimensionalityReducer(n_components=3, method="pca")
        embeddings = reducer.fit_transform(features)
    
    # 6. Cluster
    print("\n6. Clustering...")
    clusterer = ClusterAnalyzer(method="hdbscan", min_cluster_size=3)
    
    try:
        labels = clusterer.fit_predict(features)
    except Exception as e:
        print(f"   HDBSCAN failed, using K-Means: {e}")
        clusterer = ClusterAnalyzer(method="kmeans", n_clusters=5)
        labels = clusterer.fit_predict(features)
    
    stats = clusterer.get_cluster_statistics(features)
    print(f"   Found {stats['n_clusters']} clusters")
    print(f"   Noise points: {stats['n_noise_points']}")
    
    # 7. Detect patterns
    print("\n7. Detecting patterns...")
    detector = PatternDetector()
    patterns = detector.summarize_patterns(labels)
    print(f"   Unique states: {patterns['n_unique_states']}")
    
    if patterns['most_common_bigram']:
        print(f"   Most common bigram: {patterns['most_common_bigram']['pattern']} "
              f"({patterns['most_common_bigram']['count']} times)")
    
    # 8. Export results
    print("\n8. Exporting results...")
    exporter = DataExporter(output_dir=output_dir)
    paths = exporter.export_full_analysis(
        features=features,
        embeddings=embeddings,
        labels=labels,
        metadata=metadata,
        statistics=stats,
        patterns=patterns,
        filename="analysis"
    )
    
    print(f"   Exported to: {paths['data']}")
    
    # 9. Generate plots
    print("\n9. Generating plots...")
    try:
        visualizer = Visualizer(output_dir=output_dir)
        
        # 2D plot
        plot_path = visualizer.plot_embeddings_2d(
            embeddings[:, :2],
            labels,
            title="Birdsong Segments Clustering",
            filename="embeddings_2d"
        )
        print(f"   Saved 2D plot: {plot_path}")
        
        # 3D plot
        plot_path = visualizer.plot_embeddings_3d(
            embeddings,
            labels,
            title="Birdsong Segments 3D Clustering",
            filename="embeddings_3d"
        )
        print(f"   Saved 3D plot: {plot_path}")
        
        # Cluster distribution
        plot_path = visualizer.plot_cluster_distribution(
            labels,
            title="Cluster Size Distribution",
            filename="cluster_distribution"
        )
        print(f"   Saved distribution plot: {plot_path}")
        
    except ImportError:
        print("   Matplotlib not available, skipping plots")
    
    print("\n✓ Analysis complete!")
    return {
        "features": features,
        "embeddings": embeddings,
        "labels": labels,
        "metadata": metadata,
        "statistics": stats,
        "patterns": patterns,
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze birdsong audio files"
    )
    parser.add_argument(
        "audio_path",
        help="Path to audio file (WAV, MP3, FLAC, etc.)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    analyze_audio_file(args.audio_path, args.output_dir)


if __name__ == "__main__":
    main()
