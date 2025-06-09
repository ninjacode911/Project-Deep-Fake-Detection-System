# Deepfake Detection Frontend

This is the frontend application for the Deepfake Detection system. It provides a user-friendly interface for analyzing videos and detecting potential deepfakes using both visual and audio analysis.

## Features

- Video upload and analysis
- Real-time analysis progress tracking
- Visual heatmap display for detected anomalies
- Audio waveform visualization
- Detailed analysis results with confidence scores
- Support for multiple video formats
- Resource-efficient processing

## Prerequisites

- Python 3.8 or higher
- PyQt6
- NumPy
- OpenCV
- FFmpeg (for video processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deepfake-detection/frontend
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Using the Interface:
   - Click "Open Video" to select a video file
   - The video will be automatically analyzed
   - View the analysis results in the main window
   - Use the timeline to navigate through the video
   - Toggle between different visualization modes

## Configuration

The application can be configured through the `config.py` file:

- `API_BASE_URL`: Backend API endpoint
- `MAX_WORKERS`: Maximum number of concurrent analysis threads
- `CACHE_SIZE`: Size of the video frame cache
- `SUPPORTED_FORMATS`: List of supported video formats

## Troubleshooting

Common issues and solutions:

1. Video not loading:
   - Check if the video format is supported
   - Ensure FFmpeg is properly installed
   - Verify file permissions

2. Analysis not starting:
   - Check backend API connection
   - Verify video file integrity
   - Check system resources

3. Performance issues:
   - Reduce cache size in config
   - Lower the number of worker threads
   - Close other resource-intensive applications

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
