# SoundPlot: Birdsong Acoustic Analysis & Neural Synthesis Framework

[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--9548--3235-green.svg)](https://orcid.org/0009-0003-9548-3235)

An open-source framework for analyzing birdsong recordings through acoustic feature extraction, dimensionality reduction, and neural audio synthesis. Transform audio signals into interactive 3D visualizations and compare original vs synthesized acoustic trajectories in real-time.

## Project Overview

SoundPlot transforms birdsong recordings into measurable acoustic features and maps them into multi-dimensional space. By converting sound into geometry, patterns emerge that reveal the internal logic of natural soundscapes. Key capabilities:

- **Acoustic Feature Extraction**: Pitch contours, spectral centroids, MFCCs, rhythm patterns
- **Neural Audio Synthesis**: Griffin-Lim reconstruction from mel spectrograms
- **3D Visualization**: Interactive Three.js-based timbre space exploration
- **Comparative Analysis**: Side-by-side original vs synthesized waveform comparison

## Framework Statistics

| Metric | Value |
|--------|-------|
| Audio Formats Supported | WAV, MP3, FLAC, OGG, M4A |
| Sample Rate | 22.05 kHz (auto-resampled) |
| Max Audio Duration | 5 minutes (auto-trimmed) |
| Mel Correlation | 0.929 (avg synthesis quality) |
| Visualization FPS | 60 FPS (up to 5000 points) |

## Architecture

```
sound_plot/
|-- app.py                  # Flask web server
|-- ui/
|   +-- index.html          # Three.js 3D visualization
|-- src/
|   |-- audio/
|   |   |-- loader.py       # Multi-format audio I/O
|   |   |-- preprocessor.py # Normalization & segmentation
|   |   |-- synthesizer.py  # Griffin-Lim synthesis
|   |   +-- comparator.py   # Quality metrics
|   |-- features/
|   |   |-- extractor.py    # Unified feature API
|   |   |-- spectral.py     # Spectral centroid, bandwidth
|   |   |-- pitch.py        # pYIN F0 extraction
|   |   |-- mfcc.py         # MFCC computation
|   |   +-- temporal.py     # Rhythm & onset detection
|   |-- analysis/
|   |   +-- reducer.py      # PCA/UMAP reduction
|   +-- utils/
|       |-- visualization.py
|       +-- comparison.py
|-- data/
|   |-- uploads/            # User audio input
|   |-- sessions/           # Analysis output per session
|   +-- comparisons/        # Generated comparison figures
+-- publication/            # Academic paper (LaTeX)
```

## Installation

### Prerequisites
- Python 3.9+
- FFmpeg (optional, for some audio formats)

### Setup
```bash
git clone https://github.com/[your-repo]/sound_plot.git
cd sound_plot
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### Dependencies
```
librosa>=0.10.0
numpy>=1.24.0
scipy>=1.11.0
soundfile>=0.12.0
scikit-learn>=1.3.0
Flask>=2.3.0
Flask-Cors>=4.0.0
matplotlib>=3.7.0
umap-learn>=0.5.0
hdbscan>=0.8.0
```

## Quick Start

### Run the Web Server
```bash
python app.py
# Open http://localhost:5000 in your browser
```

### Web Interface Features
1. **Load Audio**: Upload WAV/MP3 files (max 16MB)
2. **View Analysis**: Dual 3D plots show original vs synthesized trajectories
3. **Play/Pause**: Independent playback for each audio stream
4. **Download**: Export audio, spectrograms, and analysis PNGs

### Run Examples
```bash
# Basic synthesis comparison
python examples/synthesis_comparison.py

# Generate birdsong sample
python generate_birdsong.py
```

## Output Structure

Each analysis session creates an organized output folder:

```
data/sessions/{audio_name}_{session_id}/
|-- original.wav           # Preprocessed input audio
|-- synthesized.wav        # Griffin-Lim reconstruction
|-- comparison.png         # Spectrogram comparison figure
|-- analysis.png           # PCA feature space visualization
+-- metadata.json          # Metrics, parameters, timestamps
```

### Quality Metrics
| Metric | Description | Typical Value |
|--------|-------------|---------------|
| SNR (dB) | Signal-to-noise ratio | -0.81 ± 0.42 |
| Waveform Corr | Time-domain correlation | ~0.00 (phase-independent) |
| Spectral Corr | STFT magnitude correlation | 0.57 ± 0.12 |
| Mel Corr | Perceptual similarity | 0.93 ± 0.04 |

## Module Details

### 1. Audio Loader (`src/audio/loader.py`)
- Loads multiple formats via librosa/soundfile
- Automatic resampling to 22.05 kHz mono
- Metadata extraction (duration, original sample rate)

### 2. Audio Preprocessor (`src/audio/preprocessor.py`)
- Min-max normalization to [-1, 1]
- Silence removal (energy-based thresholding)
- Segmentation for feature extraction

### 3. Audio Synthesizer (`src/audio/synthesizer.py`)
- Mel spectrogram extraction (128 bands)
- Griffin-Lim phase reconstruction (32 iterations)
- Post-synthesis normalization

### 4. Feature Extractor (`src/features/extractor.py`)
- Spectral: centroid, bandwidth, contrast, rolloff
- Pitch: pYIN fundamental frequency estimation
- MFCCs: 13 coefficients with deltas
- Temporal: onset detection, zero-crossing rate

### 5. Dimensionality Reducer (`src/analysis/reducer.py`)
- PCA for linear projection (fast, interpretable)
- UMAP for non-linear manifold learning
- Configurable components (2D/3D)

### 6. Comparison Visualizer (`src/utils/comparison.py`)
- Side-by-side spectrogram plots
- Feature space embedding overlays
- Automated metrics computation

## Sample Visualizations

### Spectrogram Comparison
![Comparison](data/comparisons/comparison.png)
*Original vs synthesized birdsong: waveforms, spectrograms, and mel spectrograms*

### Feature Space Analysis
![Embedding](data/comparisons/embedding_comparison.png)
*PCA projection showing drift between original (blue) and synthesized (green) features*

## Technology Stack

### Core Audio Processing
| Library | Purpose |
|---------|---------|
| **Librosa** | Feature extraction, spectrograms |
| **SoundFile** | Audio I/O (WAV, FLAC) |
| **NumPy/SciPy** | Signal processing, FFT |

### Machine Learning
| Library | Purpose |
|---------|---------|
| **Scikit-learn** | PCA, clustering |
| **UMAP-learn** | Non-linear dimensionality reduction |
| **HDBSCAN** | Density-based clustering |

### Visualization
| Library | Purpose |
|---------|---------|
| **Three.js** | WebGL 3D rendering |
| **Matplotlib** | Static figure generation |
| **Flask** | Web server backend |

## Future Enhancements

- [ ] Neural vocoder integration (HiFi-GAN, MelGAN)
- [ ] Real-time streaming analysis
- [ ] Multi-species comparison interface
- [ ] Automatic call-type clustering (HDBSCAN)
- [ ] Export to Raven selection tables

## References

- pYIN Algorithm: [Mauch & Dixon, 2014](https://www.eecs.qmul.ac.uk/~simond/pub/2014/MauchDixon-pYIN-ICASSP2014.pdf)
- Griffin-Lim: [Griffin & Lim, 1984](https://ieeexplore.ieee.org/document/1164317)
- Librosa: [McFee et al., 2015](https://librosa.org/)
- Three.js: [https://threejs.org/](https://threejs.org/)

## Data Availability

| Resource | Link |
|----------|------|
| **Source Code** | [GitHub](https://github.com/[your-repo]/sound_plot) |
| **Paper (Preprint)** | [arXiv](https://arxiv.org/) *(ID pending)* |
| **Demo Audio** | Included in `data/` directory |

## Citation

If you use SoundPlot in your research, please cite:

```bibtex
@article{mehdi2026soundplot,
  author = {Mehdi, Naqcho Ali},
  title = {SoundPlot: An Open-Source Framework for Birdsong Acoustic Analysis and Neural Synthesis},
  journal = {arXiv preprint arXiv:2026.XXXXX},
  year = {2026},
  url = {https://arxiv.org/abs/2026.XXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

© January 2026  
**Author**: Naqcho Ali Mehdi  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/naqcho-ali-mehdi-baltistani-machine-learning-engineer/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/naqchoalimehdi)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--9548--3235-green)](https://orcid.org/0009-0003-9548-3235)
