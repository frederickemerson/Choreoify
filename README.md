# Dance Movement Analysis and Synchronization

A Python-based tool for analyzing and comparing dance movements across videos using pose estimation, audio synchronization, and movement analysis.

## Features

- Real-time pose detection and tracking using YOLO
- Audio-based video synchronization
- Advanced pose comparison and analysis
- Movement sequence matching
- Detailed visual feedback and analytics
- Support for multiple dancers in the same frame
- Frame-by-frame movement analysis
- Joint angle and velocity calculations
- Depth-aware pose normalization

## Prerequisites

- Python 3.8+
- FFmpeg installed and accessible from command line
- CUDA-capable GPU (recommended for optimal performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/frederickemerson/Choreoify.git
cd dance-movement-analysis
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLO pose estimation model:
```bash
# The model will be downloaded automatically on first run
# or you can manually place 'yolo11m-pose.pt' in the project directory
```

## Usage

### Basic Usage

```python
# For simple video comparison
from dance_analysis import compare_dance_videos

compare_dance_videos('reference.mp4', 'comparison.mp4')
```

### Video Synchronization

```python
from dance_analysis import main_sync_process

# Synchronize two videos based on audio
synced_ref, synced_comp = main_sync_process('reference.mp4', 'comparison.mp4')
```

## Output

The tool generates several types of output:

1. Visual Output:
   - Side-by-side comparison video
   - Pose skeleton visualization
   - Real-time difference highlighting
   - Joint tracking visualization

2. Analysis Data:
   - JSON file with detailed pose data
   - Frame-by-frame comparison metrics
   - Movement sequence matching results
   - Joint angle and velocity analysis

## Configuration

Key parameters can be adjusted in the code:

- `model.conf`: Detection confidence threshold (default: 0.25)
- `window_size`: Comparison window size (default: 60 frames)
- `similarity_threshold`: Matching threshold (default: 0.7)
- `max_frames`: Maximum frames to process (default: 3000)

## Technical Details

### Pose Analysis
- 17-point keypoint detection
- Skeletal connection mapping
- Normalized pose comparison
- Depth-aware analysis
- Velocity and acceleration tracking

### Audio Synchronization
- Cross-correlation based alignment
- Multi-channel audio support
- Robust offset detection
- Confidence scoring

### Movement Comparison
- DTW (Dynamic Time Warping) sequence matching
- Joint angle analysis
- Velocity profiling
- Movement sequence detection

## Limitations

- Requires good lighting conditions
- Best results with clear, unobstructed views
- Performance depends on GPU capabilities
- May require manual synchronization in case of poor audio quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- YOLO for pose estimation
- OpenCV for video processing
- Librosa for audio analysis
- FFmpeg for media handling
