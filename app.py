"""
SoundPlot Web Server.

Serves the 3D visualization UI and provides an API for
analyzing and synthesizing birdsong audio.
"""

import os
import secrets
import threading
import uuid
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import librosa
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import backend modules
from src.audio import AudioLoader, AudioPreprocessor, AudioSynthesizer
from src.features import FeatureExtractor

app = Flask(__name__, static_folder="ui")
CORS(app)

# Configuration
UPLOAD_FOLDER = Path("data/uploads")
RESULTS_FOLDER = Path("data/results")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global task storage
TASKS = {}

# --- Helper Functions ---

def normalize_features(features):
    """Normalize features to 0-10 range for 3D visualization."""
    if not features:
        return []
    
    # Extract arrays
    centroids = np.array([f['centroid'] for f in features])
    bandwidths = np.array([f['bandwidth'] for f in features])
    pitches = np.array([f['pitch'] for f in features])
    
    # Helper to normalize array
    def norm(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.full_like(arr, 5.0)
        return (arr - min_val) / (max_val - min_val) * 10
    
    centroids_norm = norm(centroids)
    bandwidths_norm = norm(bandwidths)
    pitches_norm = norm(pitches)
    
    # Reassemble
    result = []
    for i, f in enumerate(features):
        result.append({
            'time': f['time'],
            'x': float(centroids_norm[i]),  # Spectral Centroid -> X
            'y': float(bandwidths_norm[i]), # Bandwidth -> Y
            'z': float(pitches_norm[i]),    # Pitch -> Z
            'raw_centroid': float(f['centroid']),
            'raw_pitch': float(f['pitch'])
        })
    return result

def run_analysis_task(task_id, file_path, save_name):
    """Heavy processing task run in a separate thread."""
    try:
        def update_status(msg, progress):
            TASKS[task_id]['status'] = msg
            TASKS[task_id]['progress'] = progress

        # 1. Load
        update_status("Loading audio file...", 5)
        loader = AudioLoader(target_sr=22050)
        audio, sr = loader.load(file_path)
        
        # 2. Enforce 5-minute limit
        MAX_DURATION = 300  # 5 minutes in seconds
        duration = len(audio) / sr
        if duration > MAX_DURATION:
            update_status(f"Trimming audio to 5 minutes (was {duration:.1f}s)...", 8)
            audio = audio[:int(MAX_DURATION * sr)]
            duration = MAX_DURATION
        
        # 3. Create session folder
        session_name = Path(file_path).stem
        session_folder = Path("data/sessions") / f"{session_name}_{task_id[:8]}"
        session_folder.mkdir(parents=True, exist_ok=True)
        
        # 4. Preprocess
        update_status("Preprocessing & Normalizing...", 15)
        preprocessor = AudioPreprocessor(sample_rate=sr)
        audio = preprocessor.normalize(audio)
        
        # Save original to session
        original_path = session_folder / "original.wav"
        sf.write(str(original_path), audio, sr)
        
        # 5. Extract features for synthesis
        update_status("Extracting features for synthesis...", 30)
        synthesizer = AudioSynthesizer(sample_rate=sr)
        synth_features = synthesizer.extract_for_synthesis(audio)
        
        # 6. Synthesize
        update_status("Neural Synthesis (Generating birdsong)...", 50)
        synthesized = synthesizer.synthesize_from_mel(synth_features["mel_spectrogram"])
        synthesized = preprocessor.normalize(synthesized)
        
        # Save synthesized to session
        synth_path = session_folder / "synthesized.wav"
        sf.write(str(synth_path), synthesized, sr)
        
        # 7. Extract time-series features for visualization
        update_status("Mapping acoustic trajectories...", 65)
        
        def extract_time_series(signal):
            """Vectorized extraction of time-series features."""
            hop_length = int(sr * 0.05) # 50ms hop
            n_fft = int(sr * 0.1) # 100ms window
            
            try:
                centroids = librosa.feature.spectral_centroid(
                    y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length
                )[0]
                
                bandwidths = librosa.feature.spectral_bandwidth(
                    y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length
                )[0]
                
                f0, voiced_flag, _ = librosa.pyin(
                    signal, fmin=100, fmax=8000, sr=sr, hop_length=hop_length
                )
                
                times = librosa.times_like(centroids, sr=sr, hop_length=hop_length)
                min_len = min(len(centroids), len(f0))
                
                chunk_features = []
                for i in range(min_len):
                    pitch = f0[i] if voiced_flag[i] else 0
                    if np.isnan(pitch): pitch = 0
                    
                    chunk_features.append({
                        'time': float(times[i]),
                        'centroid': float(centroids[i]),
                        'bandwidth': float(bandwidths[i]),
                        'pitch': float(pitch)
                    })
                return normalize_features(chunk_features)
            except Exception as e:
                print(f"Extraction error: {e}")
                import traceback
                traceback.print_exc()
                return []

        original_points = extract_time_series(audio)
        synth_points = extract_time_series(synthesized)
        
        # 8. Generate comparison visualization (spectrograms)
        update_status("Generating visualizations...", 80)
        comparison_image_url = None
        analysis_image_url = None
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for speed
            import matplotlib.pyplot as plt
            
            from src.audio import SynthesisComparator
            
            comparator = SynthesisComparator(sample_rate=sr)
            metrics = comparator.compute_comparison_metrics(audio, synthesized)
            
            # Quick spectrogram comparison plot (simplified)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Original vs Synthesized Birdsong", fontsize=14, fontweight='bold')
            
            # Waveforms
            times_orig = np.arange(len(audio)) / sr
            times_synth = np.arange(len(synthesized)) / sr
            
            axes[0, 0].plot(times_orig, audio, color='#2196F3', linewidth=0.5)
            axes[0, 0].set_title('Original Waveform')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            
            axes[0, 1].plot(times_synth, synthesized, color='#4CAF50', linewidth=0.5)
            axes[0, 1].set_title('Synthesized Waveform')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude')
            
            # Mel Spectrograms (fast computation)
            S_orig = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, hop_length=1024)
            S_synth = librosa.feature.melspectrogram(y=synthesized, sr=sr, n_mels=64, hop_length=1024)
            
            axes[1, 0].imshow(librosa.power_to_db(S_orig, ref=np.max), aspect='auto', origin='lower', cmap='magma')
            axes[1, 0].set_title('Original Mel Spectrogram')
            axes[1, 0].set_ylabel('Mel Bin')
            
            axes[1, 1].imshow(librosa.power_to_db(S_synth, ref=np.max), aspect='auto', origin='lower', cmap='magma')
            axes[1, 1].set_title('Synthesized Mel Spectrogram')
            axes[1, 1].set_ylabel('Mel Bin')
            
            # Add metrics
            metric_text = f"SNR: {metrics['snr_db']:.2f} dB | Mel Corr: {metrics['mel_correlation']:.3f}"
            fig.text(0.5, 0.02, metric_text, ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            comparison_path = session_folder / "comparison.png"
            plt.savefig(str(comparison_path), dpi=100, bbox_inches='tight')
            plt.close(fig)
            comparison_image_url = f"/data/sessions/{session_folder.name}/comparison.png"
            
            # 9. Generate feature analysis (PCA) - simplified
            update_status("Generating feature analysis...", 90)
            from src.analysis import DimensionalityReducer
            from src.features import FeatureExtractor
            
            extractor = FeatureExtractor(sample_rate=sr)
            orig_segments = preprocessor.segment(audio, 0.3, 0.15)  # Larger segments, fewer points
            synth_segments = preprocessor.segment(synthesized, 0.3, 0.15)
            
            n_segments = min(len(orig_segments), len(synth_segments), 30)  # Reduced for speed
            
            if n_segments > 5:  # Only if we have enough segments
                orig_features = []
                synth_features_list = []
                
                for i in range(n_segments):
                    orig_feat = list(extractor.extract_compact(orig_segments[i][0]).values())
                    synth_feat = list(extractor.extract_compact(synth_segments[i][0]).values())
                    orig_features.append(orig_feat)
                    synth_features_list.append(synth_feat)
                
                orig_features = np.array(orig_features)
                synth_features_arr = np.array(synth_features_list)
                
                reducer = DimensionalityReducer(n_components=2, method="pca")
                reducer.fit(orig_features)
                orig_embeddings = reducer.transform(orig_features)
                synth_embeddings = reducer.transform(synth_features_arr)
                
                # Simple embedding plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], c='#2196F3', label='Original', alpha=0.7)
                ax.scatter(synth_embeddings[:, 0], synth_embeddings[:, 1], c='#4CAF50', marker='x', label='Synthesized', alpha=0.7)
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.set_title('Feature Space: Original vs Synthesized')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                analysis_path = session_folder / "analysis.png"
                plt.savefig(str(analysis_path), dpi=100, bbox_inches='tight')
                plt.close(fig)
                analysis_image_url = f"/data/sessions/{session_folder.name}/analysis.png"
            
            # Save metadata
            metadata = {
                "original_file": save_name,
                "duration_seconds": duration,
                "sample_rate": sr,
                "metrics": metrics
            }
            with open(session_folder / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as viz_error:
            print(f"Visualization error (non-fatal): {viz_error}")
            import traceback
            traceback.print_exc()
        
        # 10. Finalize
        update_status("Complete!", 100)
        
        TASKS[task_id].update({
            "status": "Complete",
            "progress": 100,
            "result": {
                "original_points": original_points,
                "synth_points": synth_points,
                "synth_audio_url": f"/data/sessions/{session_folder.name}/synthesized.wav",
                "original_audio_url": f"/data/sessions/{session_folder.name}/original.wav",
                "comparison_image_url": comparison_image_url,
                "analysis_image_url": analysis_image_url,
                "session_folder": str(session_folder)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        TASKS[task_id]['status'] = f"Error: {str(e)}"
        TASKS[task_id]['error'] = True

# --- Routes ---

@app.route('/')
def index():
    """Serve the main UI."""
    return send_from_directory('ui', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Handle audio upload and start async analysis."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        save_name = f"{Path(filename).stem}_{secrets.token_hex(4)}{Path(filename).suffix}"
        file_path = UPLOAD_FOLDER / save_name
        file.save(str(file_path))
        
        # Create Task
        task_id = str(uuid.uuid4())
        TASKS[task_id] = {
            'status': 'Uploaded. Starting task...',
            'progress': 0,
            'result': None,
            'error': False
        }
        
        # Start Thread
        thread = threading.Thread(target=run_analysis_task, args=(task_id, file_path, save_name))
        thread.start()
        
        return jsonify({'taskId': task_id})

@app.route('/api/status/<task_id>')
def task_status(task_id):
    """Retrieve the status of a specific analysis task."""
    task = TASKS.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)

@app.route('/data/sessions/<path:filepath>')
def serve_sessions(filepath):
    return send_from_directory('data/sessions', filepath)

@app.route('/data/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/data/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting SoundPlot Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, threaded=True)
