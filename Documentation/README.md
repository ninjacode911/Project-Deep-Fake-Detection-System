# Technical Documentation: DeepGuard AI

Welcome to the internal technical documentation for the Deepfake Detection System.

## 🔬 Core Components

### 1. Model Architecture
Located in `Model Creation/`, the research phase utilized:
- **ResNext50**: Pre-trained on ImageNet, used for robust spatial feature extraction.
- **LSTM**: 2048 hidden units, processing sequences of up to 100 frames to catch temporal frame-to-frame artifacts.

### 2. Frontend Modernization (2026)
The UI has been rebuilt from scratch to provide a "Forensic Lab" feel:
- **Styles**: Custom CSS variables for neon themes located in `/static/css/styles_2026.css`.
- **Interactivity**: Alpine.js handles client-side state for the drag-and-drop zone.
- **Live Scanning**: `face-api.js` is used for browser-safe real-time face detection in the `Live Scan` module.

## ⚙️ Development Environment

### Handling Dependencies (The dlib Issue)
`dlib` and `face_recognition` can be difficult to install on Windows without C++ build tools.
- **Solution**: Use the provided `Dockerfile` in the `Django Application` folder.
- **Fallback**: The application includes a "UI-Only Mode" that allows frontend testing even if ML libraries fail to load.

## 🧪 Model Files
Ensure your trained model files are placed in `Django Application/models/`. The system dynamically searches for these based on the `Sequence Length` selected in the UI.

---
*Maintained by Navnit (AI MLOps Engineer)*
